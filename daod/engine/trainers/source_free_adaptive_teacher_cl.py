"""
Source-free teacher-student adaptation for remote sensing object detection.

This trainer implements the pipeline of Section 3.1 and Algorithm 1:

Curriculum adaptation stage (Algorithm 1, Steps 1-3)
    * CLSS ranks the target images by image-level difficulty   Eq. (3), Eq. (4)
    * the EMA teacher generates pseudo labels on weakly augmented images
    * PPBA refines the pseudo boxes inside the RoI head        Eq. (5), Eq. (6)
    * ATS filters them with a scale-adapted threshold          Eq. (7), Eq. (8)
    * the training-progress indicator p updates the baseline
      threshold of the next stage                              Eq. (9), Eq. (10)
    * the student is optimized on strongly augmented images and
      the teacher follows the student by EMA                   Eq. (2)

Fixed pseudo-label stage (Algorithm 1, Step 4)
    * AdaBN recalibrates the batch-norm statistics
    * pseudo labels are frozen and the detector is fine-tuned without EMA

Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
import copy
import json
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from fvcore.nn.precise_bn import get_bn_modules
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler

import detectron2.data.detection_utils as utils
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.data import get_detection_dataset_dicts
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase, hooks
from detectron2.engine.train_loop import AMPTrainer
from detectron2.evaluation import verify_results
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.events import EventStorage

from daod.checkpoint.detection_ts_checkpointer import DetectionTSCheckpointer
from daod.data import build_detection_semisup_train_loader_two_crops_source_free
from daod.data.build import build_semisup_batch_data_loader_two_crop_source_free
from daod.data.mappers import DatasetMapperTwoCropSeparate
from daod.engine.hooks import ValLossHook
from daod.engine.trainers import BaseTrainer
from daod.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from daod.modeling.style_transfer import StyleTransfer


class CurriculumSampler(Sampler):
    """
    Infinite sampler that only draws the target images introduced by CLSS.

    The curriculum never removes an image from the training set, it only
    decides when an image becomes visible, so the sampler iterates over the
    subset selected so far and is refreshed at every stage boundary.
    """

    def __init__(self, num_images: int):
        self.num_images = num_images
        self.selected = list(range(num_images))

    def set_selected(self, indices):
        self.selected = list(indices)

    def __iter__(self):
        while True:
            for position in torch.randperm(len(self.selected)).tolist():
                yield self.selected[position]

    def __len__(self):
        return len(self.selected)


class SourceFreeAdaptiveTeacherTrainercl(BaseTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # ------------------------------------------------------------------
        # CLSS / ATS hyper-parameters, Section 4.2
        # ------------------------------------------------------------------
        self.num_stages = cfg.SFDA.NUM_STAGES
        self.epochs_per_stage = cfg.SFDA.EPOCHS_PER_STAGE
        self.topk = cfg.SFDA.TOPK_CONF
        self.tau_loose = cfg.SFDA.TAU_LOOSE
        self.tau_init = cfg.SFDA.TAU_INIT
        self.tau_min = cfg.SFDA.TAU_MIN
        self.tau_max = cfg.SFDA.TAU_MAX
        self.phi_scale = cfg.SFDA.PHI_SCALE
        self.phi_progress = cfg.SFDA.PHI_PROGRESS
        # Baseline threshold of ATS. It starts at tau_init and is refreshed at
        # the end of every stage by Eq. (10).
        self.tau_fix = self.tau_init
        # Training-progress indicator p of Eq. (9).
        self.progress = 0.0

        # ------------------------------------------------------------------
        # CLSS state
        # ------------------------------------------------------------------
        self.image_ids = []            # target image ids, easy -> hard
        self.selected_ids = []         # images already introduced by the curriculum
        self.confidences = {}          # image id -> c(x) of the latest teacher
        self.current_stage = 0
        self.iter_per_stage = max(1, cfg.SOLVER.MAX_ITER // self.num_stages)
        self.ema_keep_rate = cfg.SEMISUPNET.EMA_KEEP_RATE

        data_loader = self._build_curriculum_loader(cfg)

        # create an student model
        print("Loading student model ...")
        model = self.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        print("Loading teacher model ...")
        model_teacher = self.build_model(cfg)
        DetectionCheckpointer(model_teacher, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    # ======================================================================
    # Data
    # ======================================================================
    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.STYLE.ENABLED:
            from daod.data.mappers import DatasetMapperEnhance

            mapper = DatasetMapperEnhance(cfg, True)
        else:
            mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops_source_free(cfg, mapper)

    def _build_curriculum_loader(self, cfg):
        """
        Build the target-domain loader whose batches are drawn from the subset
        that CLSS has introduced so far (Algorithm 1, Step 3).
        """
        dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_TARGET, filter_empty=False
        )
        self.dataset_dicts = dicts
        self.image_id_to_index = {d["image_id"]: i for i, d in enumerate(dicts)}
        self.image_ids = [d["image_id"] for d in dicts]

        if cfg.STYLE.ENABLED:
            from daod.data.mappers import DatasetMapperEnhance

            mapper = DatasetMapperEnhance(cfg, True)
        else:
            mapper = DatasetMapperTwoCropSeparate(cfg, True)

        dataset = MapDataset(DatasetFromList(dicts, copy=False), mapper)
        self.sampler = CurriculumSampler(len(dicts))
        return build_semisup_batch_data_loader_two_crop_source_free(
            dataset,
            self.sampler,
            cfg.SOLVER.IMS_PER_BATCH_TARGET,
            aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )

    # ======================================================================
    # CLSS: curriculum learning based sample sorting, Section 3.2
    # ======================================================================
    @torch.no_grad()
    def image_confidence(self, instances: Instances) -> float:
        """
        Image-level confidence c(x_i) of Eq. (3).

        Non-maximum suppression has already been applied by the detector, so
        the remaining detections are the ones above the loose threshold
        tau_loose. Only the K highest scores are averaged, which prevents a
        single over-confident background response from deciding the rank of
        an image.
        """
        scores = instances.scores
        if scores.numel() == 0:
            return 0.0
        scores = scores[scores > self.tau_loose]
        if scores.numel() == 0:
            return 0.0
        k = min(self.topk, scores.numel())
        top_scores, _ = torch.topk(scores, k)
        return top_scores.mean().item()

    def image_difficulty(self, image_id: str) -> float:
        """d(x_i) = 1 - c(x_i) of Eq. (4); a lower value means an easier image."""
        return 1.0 - self.confidences.get(image_id, 0.0)

    @torch.no_grad()
    def _teacher_confidences(self, image_ids):
        """
        Run the current EMA teacher on the given images and return c(x_i).

        The teacher predictions used here pass through PPBA inside the RoI
        head, so the difficulty of an image is estimated from the same refined
        boxes that later supervise the student.
        """
        self.model_teacher.eval()
        mapper = DatasetMapper(self.cfg, is_train=False)
        confidences = {}
        for image_id in image_ids:
            dataset_dict = self.dataset_dicts[self.image_id_to_index[image_id]]
            inputs = mapper(copy.deepcopy(dataset_dict))
            teacher_outputs = self.model_teacher([inputs], branch="unsup_data_weak")
            confidences[image_id] = self.image_confidence(teacher_outputs[2][0])
        self.model_teacher.train()
        return confidences

    def advance_curriculum(self, stage: int):
        """
        Refresh the curriculum at the beginning of a stage, Algorithm 1 Step 3.

        The images that are not selected yet are re-ranked by the current EMA
        teacher and the next group is added to the retained training set:
        q + 1 images for the first r stages and q images afterwards, with
        q = floor(N / T) and r = N mod T.
        """
        num_images = len(self.image_ids)
        q, r = divmod(num_images, self.num_stages)
        target_size = stage * q + min(stage, r)

        if not self.selected_ids:
            # Stage 1: the source-initialized teacher ranks all images once.
            self.confidences.update(self._teacher_confidences(self.image_ids))
            self.image_ids.sort(key=lambda i: -self.confidences[i])
            self.selected_ids = self.image_ids[:target_size]
        else:
            while len(self.selected_ids) < target_size:
                chosen = set(self.selected_ids)
                unselected = [i for i in self.image_ids if i not in chosen]
                self.confidences.update(self._teacher_confidences(unselected))
                unselected.sort(key=lambda i: -self.confidences[i])
                take = target_size - len(self.selected_ids)
                self.selected_ids.extend(unselected[:take])
            order = {image_id: p for p, image_id in enumerate(self.image_ids)}
            self.selected_ids.sort(key=lambda i: order[i])

        self.sampler.set_selected(
            [self.image_id_to_index[i] for i in self.selected_ids]
        )

        # Eq. (9): average image-level confidence over the current subset.
        values = [self.confidences[i] for i in self.selected_ids if i in self.confidences]
        self.progress = float(np.mean(values)) if values else 0.0
        # Eq. (10): baseline threshold of the next stage, clipped at tau_max.
        self.tau_fix = min(self.tau_init + self.phi_progress * self.progress, self.tau_max)

        logger = logging.getLogger(__name__)
        logger.info(
            "Curriculum stage %d: %d/%d images, p=%.4f, tau_fix=%.4f",
            stage, len(self.selected_ids), num_images, self.progress, self.tau_fix,
        )
        return self.selected_ids

    def dump_curriculum(self):
        """Store the curriculum order next to the checkpoints."""
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        path = os.path.join(self.cfg.OUTPUT_DIR, "curriculum_order.json")
        with open(path, "w") as handle:
            json.dump(
                {
                    "order": self.image_ids,
                    "difficulty": {i: self.image_difficulty(i) for i in self.image_ids},
                    "selected": self.selected_ids,
                    "progress": self.progress,
                    "tau_fix": self.tau_fix,
                },
                handle,
                indent=2,
            )

    # ======================================================================
    # ATS: adaptive threshold selection, Section 3.3.2
    # ======================================================================
    def ats_threshold(self, instances: Instances):
        """
        Per-instance threshold tau_i of Eq. (8).

        rho_i = a_i / a_max is the relative object area of Eq. (7): small
        instances receive a lower threshold, while tau_min prevents the
        threshold from collapsing. The confidence scores themselves are not
        modified, so ATS only decides which of the PPBA-refined boxes are
        accepted as pseudo labels.
        """
        areas = instances.pred_boxes.area()
        if areas.numel() == 0:
            return None
        rho = areas / areas.max().clamp(min=1e-6)
        thresholds = self.tau_fix - self.phi_scale * (1.0 - rho)
        return torch.clamp(thresholds, min=self.tau_min)

    # ======================================================================
    # Pseudo-label generation
    # ======================================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        """Fixed-threshold filtering, used for RPN proposals and ablations."""
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_proposal_inst.gt_boxes = Boxes(new_bbox_loc)
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_proposal_inst.gt_boxes = Boxes(new_bbox_loc)
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
        else:
            raise ValueError("Unknown proposal type: {}".format(proposal_type))

        return new_proposal_inst

    def adaptive_threshold_bbox(self, proposal_bbox_inst, proposal_type="roih"):
        """ATS filtering of Eq. (8), applied to the PPBA-refined boxes."""
        if proposal_type == "rpn":
            return self.threshold_bbox(
                proposal_bbox_inst, thres=self.tau_fix, proposal_type="rpn"
            )

        if not hasattr(proposal_bbox_inst, "pred_boxes"):
            raise AttributeError("proposal_bbox_inst must have 'pred_boxes' attribute")

        thresholds = self.ats_threshold(proposal_bbox_inst)
        if thresholds is None:
            valid_map = torch.zeros(
                (0,), dtype=torch.bool,
                device=proposal_bbox_inst.pred_boxes.tensor.device,
            )
        else:
            valid_map = proposal_bbox_inst.scores > thresholds

        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        new_proposal_inst.gt_boxes = Boxes(new_bbox_loc)
        new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
        new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
        new_proposal_inst.pred_boxes = Boxes(new_bbox_loc)
        new_proposal_inst.pred_classes = proposal_bbox_inst.pred_classes[valid_map]
        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, proposal_type, pseudo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            if pseudo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=self.tau_fix, proposal_type=proposal_type
                )
            elif pseudo_label_method == "adaptive_thresholding":
                proposal_bbox_inst = self.adaptive_threshold_bbox(
                    proposal_bbox_inst, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unknown pseudo label boxes method")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / max(len(proposals_rpn_unsup_k), 1)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        return label_list

    # ======================================================================
    # Fixed pseudo-label adaptation, Algorithm 1 Step 4
    # ======================================================================
    @torch.no_grad()
    def recalibrate_bn(self, batch_size: int = 4):
        """
        AdaBN: recompute the batch-norm statistics on the target domain while
        keeping every learned weight unchanged.
        """
        self.model.eval()
        mapper = DatasetMapper(self.cfg, is_train=False)
        buffer = []
        for image_id in self.image_ids:
            dataset_dict = self.dataset_dicts[self.image_id_to_index[image_id]]
            buffer.append(mapper(copy.deepcopy(dataset_dict)))
            if len(buffer) == batch_size:
                self.model(buffer)
                buffer = []
        if buffer:
            self.model(buffer)
        self.model.train()

    @torch.no_grad()
    def generate_frozen_pseudo_labels(self):
        """
        Generate the pseudo labels used by the fixed pseudo-label stage:
        teacher prediction -> PPBA refinement -> ATS filtering.
        """
        self.model_teacher.eval()
        mapper = DatasetMapper(self.cfg, is_train=False)
        pseudo_labels = {}
        for image_id in self.image_ids:
            dataset_dict = self.dataset_dicts[self.image_id_to_index[image_id]]
            inputs = mapper(copy.deepcopy(dataset_dict))
            teacher_outputs = self.model_teacher([inputs], branch="unsup_data_weak")
            filtered, _ = self.process_pseudo_label(
                teacher_outputs[2], "roih", "adaptive_thresholding"
            )
            pseudo_labels[image_id] = filtered[0]
        return pseudo_labels

    # ======================================================================
    # Training flow
    # ======================================================================
    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        if self.cfg.STYLE.ENABLED:
            self.style_image = utils.read_image(
                self.cfg.STYLE.STYLE_IMAGE, format=self.cfg.INPUT.FORMAT
            )
            self.style_transfer = StyleTransfer(
                self.cfg.STYLE.VGG_MODEL, self.cfg.STYLE.DECODER, self.style_image
            )

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    # CLSS: a new curriculum group is introduced whenever the
                    # stage changes, which also refreshes tau_fix through Eq. (10).
                    stage = min(self.num_stages, self.iter // self.iter_per_stage + 1)
                    if stage != self.current_stage:
                        self.current_stage = stage
                        self.advance_curriculum(stage)
                        self.storage.put_scalar("clss/stage", stage)
                        self.storage.put_scalar("clss/progress", self.progress)
                        self.storage.put_scalar("ats/tau_fix", self.tau_fix)

                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.dump_curriculum()
                self.after_train()

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def run_step(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[AdaptiveTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        unlabel_data_q, unlabel_data_k = data
        if self.cfg.STYLE.ENABLED:
            for item in unlabel_data_q:
                item["image"] = self.style_transfer(item["image"].float().to("cuda"))
        if not self.cfg.WEAK_STRONG_AUGMENT:
            unlabel_data_q = copy.deepcopy(unlabel_data_k)
        data_time = time.perf_counter() - start

        record_dict = {}

        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_k = self.remove_label(unlabel_data_k)

        # The teacher runs on weakly augmented images. PPBA is applied inside
        # its RoI head, so proposals_roih_unsup_k already holds refined boxes.
        with torch.no_grad():
            (
                _,
                proposals_rpn_unsup_k,
                proposals_roih_unsup_k,
            ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

        # CLSS statistics: image-level confidence of the current batch, Eq. (3).
        for datum, instances in zip(unlabel_data_k, proposals_roih_unsup_k):
            self.confidences[datum["image_id"]] = self.image_confidence(instances)

        # ATS, Eq. (8): filter the PPBA-refined pseudo boxes.
        pseudo_proposals_roih_unsup_k, num_pseudo_proposals_roih = self.process_pseudo_label(
            proposals_roih_unsup_k, "roih", "adaptive_thresholding"
        )
        self.storage.put_scalar(
            "roi_head/num_pseudo_proposals", num_pseudo_proposals_roih
        )

        _, num_pseudo_proposals_rpn = self.process_pseudo_label(
            proposals_rpn_unsup_k, "rpn", "thresholding"
        )
        self.storage.put_scalar("rpn/num_pseudo_proposals", num_pseudo_proposals_rpn)

        unlabel_data_q = self.add_label(unlabel_data_q, pseudo_proposals_roih_unsup_k)
        unlabel_data_k = self.add_label(unlabel_data_k, pseudo_proposals_roih_unsup_k)

        record_all_unlabel_data, predictions_roih_unsup_q, _, _ = self.model(
            unlabel_data_q, branch="supervised_target"
        )

        new_record_all_unlabel_data = {}
        for key in record_all_unlabel_data.keys():
            new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]
        record_dict.update(new_record_all_unlabel_data)

        for i_index in range(len(unlabel_data_q)):
            for k, v in unlabel_data_q[i_index].items():
                unlabel_data_k[i_index][k + "_unlabeled"] = v

        all_domain_data = unlabel_data_k
        if self.cfg.DOMAIN_CLASSIFIER.ENABLED:
            record_all_domain_data, _, _ = self.model(all_domain_data, branch="domain_classifier")
            record_dict.update(record_all_domain_data)

        # weight losses
        loss_dict = {}
        for key in record_dict.keys():
            if key.startswith("loss") and key[-3:] != "val":
                if key == "loss_box_reg_pseudo":
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                elif key[-6:] == "pseudo":  # unsupervised loss
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                elif (key == "loss_DC_img_s" or key == "loss_DC_img_t") and self.cfg.DOMAIN_CLASSIFIER.IMAGE:
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT
                elif (key == "loss_DC_ins_s" or key == "loss_DC_ins_t") and self.cfg.DOMAIN_CLASSIFIER.INSTANCE:
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT
                else:  # supervised loss
                    loss_dict[key] = record_dict[key] * 0
        losses = sum(loss_dict.values())

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        # EMA update of the teacher, Eq. (2).
        self._update_teacher_model(self.ema_keep_rate)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_teacher))
        if cfg.TEST.VAL_LOSS:
            ret.append(
                ValLossHook(
                    cfg.TEST.EVAL_PERIOD,
                    self.model,
                    build_detection_test_loader(
                        cfg, cfg.DATASETS.TEST[0],
                        mapper=DatasetMapper(self.cfg, is_train=True),
                    ),
                    model_name="_student",
                )
            )
            ret.append(
                ValLossHook(
                    cfg.TEST.EVAL_PERIOD,
                    self.model_teacher,
                    build_detection_test_loader(
                        cfg, cfg.DATASETS.TEST[0],
                        mapper=DatasetMapper(self.cfg, is_train=True),
                    ),
                )
            )

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
