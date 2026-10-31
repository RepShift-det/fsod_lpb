# """
# Adaptive teacher trainer class.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# """
# import copy
# import logging
# import time
# from collections import OrderedDict
# import numpy as np

# import detectron2.data.detection_utils as utils
# import detectron2.utils.comm as comm
# import torch
# from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase, hooks
# from detectron2.engine.train_loop import AMPTrainer
# from detectron2.utils.events import EventStorage, get_event_storage
# from detectron2.evaluation import verify_results
# from detectron2.structures.boxes import Boxes
# from detectron2.structures.instances import Instances
# from fvcore.nn.precise_bn import get_bn_modules
# from torch.nn.parallel import DistributedDataParallel
# from detectron2.data import build_detection_test_loader
# from detectron2.data.dataset_mapper import DatasetMapper
# from detectron2.checkpoint import DetectionCheckpointer
# # MetaDataCatolog.get(cfg.datasets.TRAIN).thing_classes
# from daod.checkpoint.detection_ts_checkpointer import DetectionTSCheckpointer
# from daod.data import build_detection_semisup_train_loader_two_crops_source_free
# from daod.data.mappers import DatasetMapperTwoCropSeparate
# from daod.data.mappers import DatasetMapperEnhance
# from daod.engine.trainers import BaseTrainer
# from daod.modeling.meta_arch.ts_ensemble import EnsembleTSModel
# from daod.engine.hooks import ValLossHook
# from daod.modeling.adaptive_thresh import AdaptiveConfidenceBasedSelfTrainingLoss
# from daod.modeling.style_transfer import StyleTransfer


# class SourceFreeAdaptiveTeacherTrainer(BaseTrainer):
#     def __init__(self, cfg):
#         """
#         Args:
#             cfg (CfgNode):
#         Use the custom checkpointer, which loads other backbone models
#         with matching heuristics.
#         """
#         cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
#         data_loader = self.build_train_loader(cfg)

#         # create an student model
#         print("Loading student model ...")
#         model = self.build_model(cfg)
#         DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#             cfg.MODEL.WEIGHTS, resume=False
#         )
#         optimizer = self.build_optimizer(cfg, model)
#         #print("student model device")
#         #print(model.device)

#         # create an teacher model
#         print("Loading teacher model ...")
#         model_teacher = self.build_model(cfg)
#         DetectionCheckpointer(model_teacher, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#             cfg.MODEL.WEIGHTS, resume=False
#         )       
#         self.model_teacher = model_teacher
#         #print("teacher model device")
#         #print(model_teacher.device)

#         # For training, wrap with DDP. But don't need this for inference.
#         if comm.get_world_size() > 1:
#             model = DistributedDataParallel(
#                 model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
#             )

#         TrainerBase.__init__(self)
#         self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
#             model, data_loader, optimizer
#         )
#         self.scheduler = self.build_lr_scheduler(cfg, optimizer)

#         # Ensemble teacher and student model is for model saving and loading
#         ensem_ts_model = EnsembleTSModel(model_teacher, model)

#         self.checkpointer = DetectionTSCheckpointer(
#             ensem_ts_model,
#             cfg.OUTPUT_DIR,
#             optimizer=optimizer,
#             scheduler=self.scheduler,
#         )
#         self.start_iter = 0
#         self.max_iter = cfg.SOLVER.MAX_ITER
#         self.cfg = cfg

#         self.register_hooks(self.build_hooks())

#     @classmethod
#     def build_train_loader(cls, cfg):
#         if cfg.STYLE.ENABLED:
#             mapper = DatasetMapperEnhance(cfg, True)        
#         else:
#             mapper = DatasetMapperTwoCropSeparate(cfg, True)
            
#         return build_detection_semisup_train_loader_two_crops_source_free(cfg, mapper)
    
#     def train_loop(self, start_iter: int, max_iter: int):
#         """
#         Args:
#             start_iter, max_iter (int): See docs above
#         """
#         logger = logging.getLogger(__name__)
#         logger.info("Starting training from iteration {}".format(start_iter))

#         self.iter = self.start_iter = start_iter
#         self.max_iter = max_iter

#         if self.cfg.ADAPTIVE_THRESHOLD.ENABLED:
#             # initialize adaptive threshold 
#             fix_thresh = self.cfg.SEMISUPNET.BBOX_THRESHOLD
#             self.self_training_criterion = AdaptiveConfidenceBasedSelfTrainingLoss(threshold=fix_thresh, num_classes=self.cfg.MODEL.ROI_HEADS.NUM_CLASSES)
#             self.reserve_matrix = torch.zeros((self.cfg.ADAPTIVE_THRESHOLD.RESERVE, self.cfg.MODEL.ROI_HEADS.NUM_CLASSES)).cuda()

#         if self.cfg.STYLE.ENABLED:
#             self.style_image = utils.read_image(self.cfg.STYLE.STYLE_IMAGE, format=self.cfg.INPUT.FORMAT)
#             self.style_transfer = StyleTransfer(self.cfg.STYLE.VGG_MODEL, self.cfg.STYLE.DECODER, self.style_image)
        
#         with EventStorage(start_iter) as self.storage:
#             try:
#                 self.before_train()
#                 for self.iter in range(start_iter, max_iter):
#                     self.before_step()
#                     self.run_step()
#                     #print("Before after step: ")
#                     self.after_step()
#                     #print("After after step ")
#             except Exception:
#                 logger.exception("Exception during training:")
#                 raise
#             finally:
#                 self.after_train()

#     def train(self):
#         self.train_loop(self.start_iter, self.max_iter)
#         if hasattr(self, "_last_eval_results") and comm.is_main_process():
#             verify_results(self.cfg, self._last_eval_results)
#             return self._last_eval_results
#     def compute_dynamic_threshold(self, proposal_bbox_inst, thres):
#         # get the confidence scores predicted by the model
#         scores = proposal_bbox_inst.scores

#         # compute the object area
#         bbox_area = proposal_bbox_inst.pred_boxes.area()

#         # check whether bbox_area is empty
#         if bbox_area.numel() == 0:
#             # if no object is detected, set difficulty_weight to 1.0, i.e. the highest difficulty
#             difficulty_weight = torch.ones_like(scores)
#         else:
#             # normalise the object area: the smaller the area, the higher the difficulty
#             difficulty_weight = bbox_area / bbox_area.max()  # normalise the object area

#         # adaptive threshold: driven by object difficulty and confidence
#         # the higher the difficulty or the lower the confidence, the lower the threshold
#         adjusted_threshold = thres - 0.2 * difficulty_weight

#         # keep the threshold within a valid range
#         adjusted_threshold = torch.clamp(adjusted_threshold, min=0.3, max=1.0)

#         return adjusted_threshold

#     # =====================================================
#     # ================== Pseudo-labeling ==================
#     # =====================================================
#     def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
#         if proposal_type == "rpn":
#             valid_map = proposal_bbox_inst.objectness_logits > thres

#             # create instances containing boxes and gt_classes
#             image_shape = proposal_bbox_inst.image_size
#             new_proposal_inst = Instances(image_shape)

#             # create box
#             new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
#             new_boxes = Boxes(new_bbox_loc)

#             # add boxes to instances
#             new_proposal_inst.gt_boxes = new_boxes
#             new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
#                 valid_map
#             ]
#         elif proposal_type == "roih":
#             valid_map = proposal_bbox_inst.scores > thres

#             # create instances containing boxes and gt_classes
#             image_shape = proposal_bbox_inst.image_size
#             new_proposal_inst = Instances(image_shape)

#             # create box
#             new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
#             new_boxes = Boxes(new_bbox_loc)

#             # add boxes to instances
#             new_proposal_inst.gt_boxes = new_boxes
#             new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
#             new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

#         return new_proposal_inst

#     def adaptive_threshold_bbox(self, proposal_bbox_inst, thres, proposal_type="roih"):
#         if proposal_type == "rpn":
#             valid_map = proposal_bbox_inst.objectness_logits > thres

#             # create instances containing boxes and gt_classes
#             image_shape = proposal_bbox_inst.image_size
#             new_proposal_inst = Instances(image_shape)

#             # create box
#             new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
#             new_boxes = Boxes(new_bbox_loc)

#             # add boxes to instances
#             new_proposal_inst.gt_boxes = new_boxes
#             new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[valid_map]
#         elif proposal_type == "roih":
#             if not hasattr(proposal_bbox_inst, 'pred_boxes'):
#                 raise AttributeError("proposal_bbox_inst must have 'pred_boxes' attribute")

#             # compute the adaptive threshold
#             dynamic_thresholds = self.compute_dynamic_threshold(proposal_bbox_inst, thres)
#             valid_map = proposal_bbox_inst.scores > dynamic_thresholds

#             # create instances containing boxes and gt_classes
#             image_shape = proposal_bbox_inst.image_size
#             new_proposal_inst = Instances(image_shape)

#             # create box
#             new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
#             new_boxes = Boxes(new_bbox_loc)

#             # add boxes to instances
#             new_proposal_inst.gt_boxes = new_boxes
#             new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
#             new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

#         return new_proposal_inst

#     def prediction_threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
#         if proposal_type == "roih":
#             prediction_classes = proposal_bbox_inst.pred_classes
#             prediction_scores = proposal_bbox_inst.scores
#             valid_map = self.self_training_criterion(prediction_scores, prediction_classes)

#             valid_map = torch.Tensor((np.where(valid_map.cpu().numpy() == 1)[0])).cuda()
#             valid_map = valid_map.long()

#             # print(valid_map)

#             # create instances containing boxes and gt_classes
#             image_shape = proposal_bbox_inst.image_size
#             new_proposal_inst = Instances(image_shape)

#             # create box
#             new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
#             new_boxes = Boxes(new_bbox_loc)

#             # add boxes to instances
#             new_proposal_inst.pred_boxes = new_boxes
#             new_proposal_inst.pred_classes = proposal_bbox_inst.pred_classes[valid_map]
#             new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

#         return new_proposal_inst

#     def process_pseudo_label(
#         self, proposals_rpn_unsup_k, cur_threshold, proposal_type, pseudo_label_method=""
#     ):
#         list_instances = []
#         num_proposal_output = 0.0
#         for proposal_bbox_inst in proposals_rpn_unsup_k:
#             # thresholding
#             if pseudo_label_method == "thresholding":
#                 proposal_bbox_inst = self.threshold_bbox(
#                     proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
#                 )
#             elif pseudo_label_method == "adaptive_thresholding":
#                 proposal_bbox_inst = self.adaptive_threshold_bbox(
#                     proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
#                 )
#             elif pseudo_label_method == "prediction_thresholding":
#                 proposal_bbox_inst = self.prediction_threshold_bbox(
#                     proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
#                 )           
#             else:
#                 raise ValueError("Unknown pseudo label boxes method")
#             num_proposal_output += len(proposal_bbox_inst)
#             list_instances.append(proposal_bbox_inst)
#         num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
#         return list_instances, num_proposal_output


# # todo
#     # def count_label_prediction(self, predictions_roih_unsup_q):
#     #     # the reserve count will update the reserve matrix for one batch
#     #     # the shape of reserve count is C
#     #     # B is the batch size and 
#     #     reserve_count = torch.zeros(self.cfg.MODEL.ROI_HEADS.NUM_CLASSES).cuda()
#     #     for prediction_bbox_inst in predictions_roih_unsup_q:
#     #         # for each item in the batch
#     #         # print(prediction_bbox_inst)
#     #         prediction_classes = prediction_bbox_inst.pred_classes
#     #         prediction_scores = prediction_bbox_inst.scores  
#     #         valid_map = prediction_scores > self.cfg.SEMISUPNET.BBOX_THRESHOLD
#     #         reserve_classes = prediction_classes[valid_map]
#     #         reserve_count += reserve_classes.bincount(minlength=self.cfg.MODEL.ROI_HEADS.NUM_CLASSES)
#     #     return reserve_count
    
#     def count_label_prediction(self, predictions_roih_unsup_q):
#         if self.cfg.MODEL.ROI_HEADS.NUM_CLASSES == 1:
#             reserve_count = torch.zeros(1).cuda()
#             for prediction_bbox_inst in predictions_roih_unsup_q:
#                 prediction_scores = prediction_bbox_inst.scores
#                 valid_map = prediction_scores > self.cfg.SEMISUPNET.BBOX_THRESHOLD
#                 reserve_count[0] += valid_map.sum().item()
#             return reserve_count
#         else:
#             reserve_count = torch.zeros(self.cfg.MODEL.ROI_HEADS.NUM_CLASSES).cuda()
#             for prediction_bbox_inst in predictions_roih_unsup_q:
#                 prediction_classes = prediction_bbox_inst.pred_classes
#                 prediction_scores = prediction_bbox_inst.scores
#                 valid_map = prediction_scores > self.cfg.SEMISUPNET.BBOX_THRESHOLD
#                 reserve_classes = prediction_classes[valid_map]
#                 reserve_count += reserve_classes.bincount(minlength=self.cfg.MODEL.ROI_HEADS.NUM_CLASSES)
#             return reserve_count
# # todo
#     # def update_adaptive_threshold(self):
#     #     # self.reserve_matrix will record previous class prediction results
#     #     # the shape of reserve_matrix is N * C
#     #     # N is the number of batches, C is the number of classes
#     #     classwise_counter = self.reserve_matrix.sum(dim=0)
        
#     #     # Ensure classwise_counter has the correct size
#     #     if classwise_counter.size(0) != self.cfg.MODEL.ROI_HEADS.NUM_CLASSES:
#     #         raise ValueError(f"Expected classwise_counter size to be {self.cfg.MODEL.ROI_HEADS.NUM_CLASSES}, but got {classwise_counter.size(0)}")

#     #     # Avoid division by zero
#     #     max_counter = max(classwise_counter.max(), 1)
        
#     #     # Update classwise accuracy
#     #     self.self_training_criterion.classwise_acc = classwise_counter / max_counter

#     #     # Ensure classwise_acc has the correct size
#     #     if self.self_training_criterion.classwise_acc.size(0) != self.cfg.MODEL.ROI_HEADS.NUM_CLASSES:
#     #         raise ValueError(f"Expected classwise_acc size to be {self.cfg.MODEL.ROI_HEADS.NUM_CLASSES}, but got {self.self_training_criterion.classwise_acc.size(0)}")

#     #     # Log classwise accuracy
#     #     for i in range(self.cfg.MODEL.ROI_HEADS.NUM_CLASSES):
#     #         self.storage.put_scalar(
#     #             "acc_thres/class_" + str(i), self.self_training_criterion.classwise_acc[i]
#     #         )
#     def update_adaptive_threshold(self):
#         classwise_counter = self.reserve_matrix.sum(dim=0)
#         if self.cfg.MODEL.ROI_HEADS.NUM_CLASSES == 1:
#             max_counter = max(classwise_counter[0].item(), 1)
#             self.self_training_criterion.classwise_acc = classwise_counter / max_counter
#             self.storage.put_scalar(
#                 "acc_thres/class_0", self.self_training_criterion.classwise_acc[0]
#             )
#         else:
#             if classwise_counter.size(0) != self.cfg.MODEL.ROI_HEADS.NUM_CLASSES:
#                 raise ValueError(f"Expected classwise_counter size to be {self.cfg.MODEL.ROI_HEADS.NUM_CLASSES}, but got {classwise_counter.size(0)}")
#             max_counter = max(classwise_counter.max(), 1)
#             self.self_training_criterion.classwise_acc = classwise_counter / max_counter
#             if self.self_training_criterion.classwise_acc.size(0) != self.cfg.MODEL.ROI_HEADS.NUM_CLASSES:
#                 raise ValueError(f"Expected classwise_acc size to be {self.cfg.MODEL.ROI_HEADS.NUM_CLASSES}, but got {self.self_training_criterion.classwise_acc.size(0)}")
#             for i in range(self.cfg.MODEL.ROI_HEADS.NUM_CLASSES):
#                 self.storage.put_scalar(
#                     "acc_thres/class_" + str(i), self.self_training_criterion.classwise_acc[i]
#                 )

#     def remove_label(self, label_data):
#         for label_datum in label_data:
#             if "instances" in label_datum.keys():
#                 del label_datum["instances"]
#         return label_data

#     def add_label(self, unlabled_data, label):
#         for unlabel_datum, lab_inst in zip(unlabled_data, label):
#             unlabel_datum["instances"] = lab_inst
#         return unlabled_data
    
#     def get_label(self, label_data):
#         label_list = []
#         for label_datum in label_data:
#             if "instances" in label_datum.keys():
#                 label_list.append(copy.deepcopy(label_datum["instances"]))
#         return label_list

#     # =====================================================
#     # =================== Training Flow ===================
#     # =====================================================

#     def run_step(self):
#         self._trainer.iter = self.iter
#         assert self.model.training, "[AdaptiveTeacherTrainer] model was changed to eval mode!"
#         start = time.perf_counter()
#         data = next(self._trainer._data_loader_iter)
#         unlabel_data_q, unlabel_data_k = data
#         if self.cfg.STYLE.ENABLED:
#             for item in unlabel_data_q:
#                 item['image'] = self.style_transfer(item['image'].float().to("cuda"))
#         if not self.cfg.WEAK_STRONG_AUGMENT:
#             unlabel_data_q = copy.deepcopy(unlabel_data_k)
#         data_time = time.perf_counter() - start

#         record_dict = {}

#         unlabel_data_q = self.remove_label(unlabel_data_q)
#         unlabel_data_k = self.remove_label(unlabel_data_k)

#         with torch.no_grad():
#             (
#                 _,
#                 proposals_rpn_unsup_k,
#                 proposals_roih_unsup_k
#             ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

#         if self.cfg.ADAPTIVE_THRESHOLD.ENABLED:
#             proposals_roih_unsup_k_thres, _ = self.process_pseudo_label(
#                 proposals_roih_unsup_k, self.cfg.SEMISUPNET.BBOX_THRESHOLD, "roih", "prediction_thresholding"
#             )
#             self.reserve_matrix[self.iter % self.cfg.ADAPTIVE_THRESHOLD.RESERVE] = self.count_label_prediction(proposals_roih_unsup_k_thres)
#             self.update_adaptive_threshold()
#             for i in range(self.cfg.MODEL.ROI_HEADS.NUM_CLASSES):
#                 self.storage.put_scalar(
#                     "acc_thres/class_" + str(i), self.self_training_criterion.classwise_acc[i]
#                 )

#         confidences = np.zeros(len(proposals_roih_unsup_k))
#         for i, proposal_bbox_inst in enumerate(proposals_roih_unsup_k):
#             prediction_scores = proposal_bbox_inst.scores
#             confidences[i] = prediction_scores.mean().item()
#         mean_confidence = confidences.mean()
#         self.storage.put_scalar(
#             "roi_head/mean_confidence", mean_confidence
#         )

#         cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

#         joint_proposal_dict = {}
#         joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
#         pseudo_proposals_rpn_unsup_k, num_pseudo_proposals_rpn = self.process_pseudo_label(
#             proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
#         )
#         self.storage.put_scalar(
#             "rpn/num_pseudo_proposals", num_pseudo_proposals_rpn
#         )
#         joint_proposal_dict["proposals_pseudo_rpn"] = pseudo_proposals_rpn_unsup_k

#         if self.cfg.ADAPTIVE_THRESHOLD.ENABLED and self.iter >= self.cfg.ADAPTIVE_THRESHOLD.WARM_UP:
#             pseudo_proposals_roih_unsup_k, num_pseudo_proposals_roih = self.process_pseudo_label(
#                 proposals_roih_unsup_k, cur_threshold, "roih", "adaptive_thresholding"
#             )
#         else:
#             pseudo_proposals_roih_unsup_k, num_pseudo_proposals_roih = self.process_pseudo_label(
#                 proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
#             )
#         joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_roih_unsup_k
#         self.storage.put_scalar(
#             "roi_head/num_pseudo_proposals", num_pseudo_proposals_roih
#         )

#         unlabel_data_q = self.add_label(
#             unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
#         )
#         unlabel_data_k = self.add_label(
#             unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
#         )

#         all_unlabel_data = unlabel_data_q

#         record_all_unlabel_data, predictions_roih_unsup_q, _, _ = self.model(
#             all_unlabel_data, branch="supervised_target"
#         )

#         new_record_all_unlabel_data = {}
#         for key in record_all_unlabel_data.keys():
#             new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]
#         record_dict.update(new_record_all_unlabel_data)

#         for i_index in range(len(unlabel_data_q)):
#             for k, v in unlabel_data_q[i_index].items():
#                 unlabel_data_k[i_index][k + "_unlabeled"] = v

#         all_domain_data = unlabel_data_k
#         if self.cfg.DOMAIN_CLASSIFIER.ENABLED:
#             record_all_domain_data, _, _ = self.model(all_domain_data, branch="domain_classifier")
#             record_dict.update(record_all_domain_data)

#         # weight losses
#         loss_dict = {}
#         for key in record_dict.keys():
#             #print(key)
#             #print(record_dict[key])
#             if key.startswith("loss") and key[-3:] != "val":
#                 if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
#                     # pseudo bbox regression <- 0
#                     #loss_dict[key] = record_dict[key] * 0
#                     loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
#                 elif key == "loss_bpc_pseudo":
#                     loss_dict[key] = record_dict[key] * 0
#                 elif key[-6:] == "pseudo":  # unsupervised loss
#                     loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
#                     #print(key)
#                     #print(loss_dict[key])
#                 elif (key == "loss_DC_img_s" or key == "loss_DC_img_t") and self.cfg.DOMAIN_CLASSIFIER.IMAGE:  # set weight for discriminator
#                     loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT #Need to modify defaults and yaml
#                 elif (key == "loss_DC_ins_s" or key == "loss_DC_ins_t") and self.cfg.DOMAIN_CLASSIFIER.INSTANCE:  # set weight for discriminator
#                     loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT #Need to modify defaults and yaml
#                 elif (key == "loss_bpc_pseudo"):
#                     loss_dict[key] = record_dict[key] * 0
#                 else:  # supervised loss
#                     #print(key)
#                     #print(record_dict[key])
#                     loss_dict[key] = record_dict[key] * 0
#         #print("-------------------------------------------------------------")
#         self.storage.put_scalar(
#             "calibration/bpc_loss", loss_dict["loss_bpc_pseudo"]
#         )
#         losses = sum(loss_dict.values())
#         #print(loss_dict)
#         #print(losses)

#         #metrics_dict = record_dict
#         metrics_dict = loss_dict
#         metrics_dict["data_time"] = data_time
#         self._write_metrics(metrics_dict)

#         self.optimizer.zero_grad()
#         losses.backward()
#         self.optimizer.step()
#         #self._update_teacher_model()

#     @torch.no_grad()
#     def _update_teacher_model(self, keep_rate=0.9996):
#         #keep_rate = 0.999
#         if comm.get_world_size() > 1:
#             student_model_dict = {
#                 key[7:]: value for key, value in self.model.state_dict().items()
#             }
#         else:
#             student_model_dict = self.model.state_dict()

#         new_teacher_dict = OrderedDict()
#         for key, value in self.model_teacher.state_dict().items():
#             if key in student_model_dict.keys():
#                 new_teacher_dict[key] = (
#                     student_model_dict[key] *
#                     (1 - keep_rate) + value * keep_rate
#                 )
#             else:
#                 raise Exception("{} is not found in student model".format(key))

#         self.model_teacher.load_state_dict(new_teacher_dict)

#     @torch.no_grad()
#     def _copy_main_model(self):
#         # initialize all parameters
#         if comm.get_world_size() > 1:
#             rename_model_dict = {
#                 key[7:]: value for key, value in self.model.state_dict().items()
#             }
#             self.model_teacher.load_state_dict(rename_model_dict)
#         else:
#             self.model_teacher.load_state_dict(self.model.state_dict())

#     def build_hooks(self):
#         cfg = self.cfg.clone()
#         cfg.defrost()
#         cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

#         ret = [
#             hooks.IterationTimer(),
#             hooks.LRScheduler(self.optimizer, self.scheduler),
#             hooks.PreciseBN(
#                 # Run at the same freq as (but before) evaluation.
#                 cfg.TEST.EVAL_PERIOD,
#                 self.model,
#                 # Build a new data loader to not affect training
#                 self.build_train_loader(cfg),
#                 cfg.TEST.PRECISE_BN.NUM_ITER,
#             )
#             if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
#             else None,
#         ]

#         # Do PreciseBN before checkpointer, because it updates the model and need to
#         # be saved by checkpointer.
#         # This is not always the best: if checkpointing has a different frequency,
#         # some checkpoints may have more precise statistics than others.
#         if comm.is_main_process():
#             ret.append(
#                 hooks.PeriodicCheckpointer(
#                     self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
#                 )
#             )

#         def test_and_save_results_student():
#             self._last_eval_results_student = self.test(self.cfg, self.model)
#             _last_eval_results_student = {
#                 k + "_student": self._last_eval_results_student[k]
#                 for k in self._last_eval_results_student.keys()
#             }
#             return _last_eval_results_student

#         def test_and_save_results_teacher():
#             self._last_eval_results_teacher = self.test(
#                 self.cfg, self.model_teacher)
#             return self._last_eval_results_teacher

#         ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
#                    test_and_save_results_student))
#         ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
#                    test_and_save_results_teacher))
#         if cfg.TEST.VAL_LOSS:
#             ret.append(ValLossHook(
#                 cfg.TEST.EVAL_PERIOD,
#                 self.model,
#                 build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=DatasetMapper(self.cfg, is_train=True)),
#                 model_name="_student"
#             ))
#             ret.append(ValLossHook(
#                 cfg.TEST.EVAL_PERIOD,
#                 self.model_teacher,
#                 build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=DatasetMapper(self.cfg, is_train=True))
#             ))

#         if comm.is_main_process():
#             # run writers in the end, so that evaluation metrics are written
#             ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
#         return ret


import copy
import logging
import time
from collections import OrderedDict
import numpy as np

import detectron2.data.detection_utils as utils
import detectron2.utils.comm as comm
import torch
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase, hooks
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.evaluation import verify_results
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from fvcore.nn.precise_bn import get_bn_modules
from torch.nn.parallel import DistributedDataParallel
from detectron2.data import build_detection_test_loader
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.checkpoint import DetectionCheckpointer
# MetaDataCatolog.get(cfg.datasets.TRAIN).thing_classes
from daod.checkpoint.detection_ts_checkpointer import DetectionTSCheckpointer
from daod.data import build_detection_semisup_train_loader_two_crops_source_free
from daod.data.mappers import DatasetMapperTwoCropSeparate
from daod.data.mappers import DatasetMapperEnhance
from daod.engine.trainers import BaseTrainer
from daod.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from daod.engine.hooks import ValLossHook
from daod.modeling.adaptive_thresh import AdaptiveConfidenceBasedSelfTrainingLoss
from daod.modeling.style_transfer import StyleTransfer


class SourceFreeAdaptiveTeacherTrainer(BaseTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

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

        # initialise the base threshold, the control coefficients and the lower bound
        self.theta_fix = 0.7
        self.theta_initial = 0.7
        self.alpha = cfg.ADAPTIVE_THRESHOLD.ALPHA
        self.theta_min = 0.3
        self.gamma = cfg.ADAPTIVE_THRESHOLD.GAMMA

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.STYLE.ENABLED:
            mapper = DatasetMapperEnhance(cfg, True)
        else:
            mapper = DatasetMapperTwoCropSeparate(cfg, True)

        return build_detection_semisup_train_loader_two_crops_source_free(cfg, mapper)

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        if self.cfg.ADAPTIVE_THRESHOLD.ENABLED:
            # initialize adaptive threshold
            fix_thresh = self.cfg.SEMISUPNET.BBOX_THRESHOLD
            self.self_training_criterion = AdaptiveConfidenceBasedSelfTrainingLoss(threshold=fix_thresh,
                                                                                   num_classes=self.cfg.MODEL.ROI_HEADS.NUM_CLASSES)
            self.reserve_matrix = torch.zeros((self.cfg.ADAPTIVE_THRESHOLD.RESERVE,
                                               self.cfg.MODEL.ROI_HEADS.NUM_CLASSES)).cuda()

        if self.cfg.STYLE.ENABLED:
            self.style_image = utils.read_image(self.cfg.STYLE.STYLE_IMAGE, format=self.cfg.INPUT.FORMAT)
            self.style_transfer = StyleTransfer(self.cfg.STYLE.VGG_MODEL, self.cfg.STYLE.DECODER, self.style_image)

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def compute_dynamic_threshold(self, proposal_bbox_inst):
        # object box area
        a = proposal_bbox_inst.pred_boxes.area()
        if a.numel() == 0:
            # if no object is detected, set the difficulty weight to 1.0
            w = torch.ones_like(proposal_bbox_inst.scores)
        else:
            # maximum area
            a_max = a.max()
            # difficulty weight
            w = a / a_max

        # adaptive threshold derived from the difficulty weight
        theta = torch.max(self.theta_fix - self.alpha * w, torch.tensor(self.theta_min).to(w.device))
        return theta

    # =====================================================
    # ================== Pseudo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def adaptive_threshold_bbox(self, proposal_bbox_inst, proposal_type="roih"):
        if proposal_type == "rpn":
            # the RPN does not use the adaptive threshold; keep the original logic
            thres = self.theta_fix
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[valid_map]
        elif proposal_type == "roih":
            if not hasattr(proposal_bbox_inst, 'pred_boxes'):
                raise AttributeError("proposal_bbox_inst must have 'pred_boxes' attribute")

            # compute the adaptive threshold
            dynamic_thresholds = self.compute_dynamic_threshold(proposal_bbox_inst)
            valid_map = proposal_bbox_inst.scores > dynamic_thresholds

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def prediction_threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "roih":
            prediction_classes = proposal_bbox_inst.pred_classes
            prediction_scores = proposal_bbox_inst.scores
            valid_map = self.self_training_criterion(prediction_scores, prediction_classes)

            valid_map = torch.Tensor((np.where(valid_map.cpu().numpy() == 1)[0])).cuda()
            valid_map = valid_map.long()

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.pred_boxes = new_boxes
            new_proposal_inst.pred_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
            self, proposals_rpn_unsup_k, proposal_type, pseudo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if pseudo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=self.theta_fix, proposal_type=proposal_type
                )
            elif pseudo_label_method == "adaptive_thresholding":
                proposal_bbox_inst = self.adaptive_threshold_bbox(
                    proposal_bbox_inst, proposal_type=proposal_type
                )
            elif pseudo_label_method == "prediction_thresholding":
                proposal_bbox_inst = self.prediction_threshold_bbox(
                    proposal_bbox_inst, thres=self.theta_fix, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unknown pseudo label boxes method")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def count_label_prediction(self, predictions_roih_unsup_q):
        if self.cfg.MODEL.ROI_HEADS.NUM_CLASSES == 1:
            reserve_count = torch.zeros(1).cuda()
            for prediction_bbox_inst in predictions_roih_unsup_q:
                prediction_scores = prediction_bbox_inst.scores
                valid_map = prediction_scores > self.theta_fix
                reserve_count[0] += valid_map.sum().item()
            return reserve_count
        else:
            reserve_count = torch.zeros(self.cfg.MODEL.ROI_HEADS.NUM_CLASSES).cuda()
            for prediction_bbox_inst in predictions_roih_unsup_q:
                prediction_classes = prediction_bbox_inst.pred_classes
                prediction_scores = prediction_bbox_inst.scores
                valid_map = prediction_scores > self.theta_fix
                reserve_classes = prediction_classes[valid_map]
                reserve_count += reserve_classes.bincount(minlength=self.cfg.MODEL.ROI_HEADS.NUM_CLASSES)
            return reserve_count

    def update_adaptive_threshold(self):
        classwise_counter = self.reserve_matrix.sum(dim=0)
        if self.cfg.MODEL.ROI_HEADS.NUM_CLASSES == 1:
            max_counter = max(classwise_counter[0].item(), 1)
            acc = classwise_counter / max_counter
            self.self_training_criterion.classwise_acc = acc
            self.storage.put_scalar(
                "acc_thres/class_0", self.self_training_criterion.classwise_acc[0]
            )
            # update the base threshold
            self.theta_fix = self.theta_initial + self.gamma * acc[0].item()
            self.storage.put_scalar(
                "theta_fix", self.theta_fix
            )
        else:
            if classwise_counter.size(0) != self.cfg.MODEL.ROI_HEADS.NUM_CLASSES:
                raise ValueError(
                    f"Expected classwise_counter size to be {self.cfg.MODEL.ROI_HEADS.NUM_CLASSES}, but got {classwise_counter.size(0)}")
            max_counter = max(classwise_counter.max(), 1)
            acc = classwise_counter / max_counter
            self.self_training_criterion.classwise_acc = acc
            if self.self_training_criterion.classwise_acc.size(0) != self.cfg.MODEL.ROI_HEADS.NUM_CLASSES:
                raise ValueError(
                    f"Expected classwise_acc size to be {self.cfg.MODEL.ROI_HEADS.NUM_CLASSES}, but got {self.self_training_criterion.classwise_acc.size(0)}")
            for i in range(self.cfg.MODEL.ROI_HEADS.NUM_CLASSES):
                self.storage.put_scalar(
                    "acc_thres/class_" + str(i), self.self_training_criterion.classwise_acc[i]
                )
            # update the base threshold with the mean accuracy over all classes
            avg_acc = acc.mean().item()
            self.theta_fix = self.theta_initial + self.gamma * avg_acc
            self.storage.put_scalar(
                "theta_fix", self.theta_fix
            )

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

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[AdaptiveTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        unlabel_data_q, unlabel_data_k = data
        if self.cfg.STYLE.ENABLED:
            for item in unlabel_data_q:
                item['image'] = self.style_transfer(item['image'].float().to("cuda"))
        if not self.cfg.WEAK_STRONG_AUGMENT:
            unlabel_data_q = copy.deepcopy(unlabel_data_k)
        data_time = time.perf_counter() - start

        record_dict = {}

        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_k = self.remove_label(unlabel_data_k)

        with torch.no_grad():
            (
                _,
                proposals_rpn_unsup_k,
                proposals_roih_unsup_k
            ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

        if self.cfg.ADAPTIVE_THRESHOLD.ENABLED:
            proposals_roih_unsup_k_thres, _ = self.process_pseudo_label(
                proposals_roih_unsup_k, "roih", "prediction_thresholding"
            )
            self.reserve_matrix[self.iter % self.cfg.ADAPTIVE_THRESHOLD.RESERVE] = self.count_label_prediction(
                proposals_roih_unsup_k_thres)
            self.update_adaptive_threshold()
            for i in range(self.cfg.MODEL.ROI_HEADS.NUM_CLASSES):
                self.storage.put_scalar(
                    "acc_thres/class_" + str(i), self.self_training_criterion.classwise_acc[i]
                )

        confidences = np.zeros(len(proposals_roih_unsup_k))
        for i, proposal_bbox_inst in enumerate(proposals_roih_unsup_k):
            prediction_scores = proposal_bbox_inst.scores
            confidences[i] = prediction_scores.mean().item()
        mean_confidence = confidences.mean()
        self.storage.put_scalar(
            "roi_head/mean_confidence", mean_confidence
        )

        cur_threshold = self.theta_fix

        joint_proposal_dict = {}
        joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
        pseudo_proposals_rpn_unsup_k, num_pseudo_proposals_rpn = self.process_pseudo_label(
            proposals_rpn_unsup_k, "rpn", "thresholding"
        )
        self.storage.put_scalar(
            "rpn/num_pseudo_proposals", num_pseudo_proposals_rpn
        )
        joint_proposal_dict["proposals_pseudo_rpn"] = pseudo_proposals_rpn_unsup_k

        if self.cfg.ADAPTIVE_THRESHOLD.ENABLED and self.iter >= self.cfg.ADAPTIVE_THRESHOLD.WARM_UP:
            pseudo_proposals_roih_unsup_k, num_pseudo_proposals_roih = self.process_pseudo_label(
                proposals_roih_unsup_k, "roih", "adaptive_thresholding"
            )
        else:
            pseudo_proposals_roih_unsup_k, num_pseudo_proposals_roih = self.process_pseudo_label(
                proposals_roih_unsup_k, "roih", "thresholding"
            )
        joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_roih_unsup_k
        self.storage.put_scalar(
            "roi_head/num_pseudo_proposals", num_pseudo_proposals_roih
        )

        unlabel_data_q = self.add_label(
            unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
        )
        unlabel_data_k = self.add_label(
            unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
        )

        all_unlabel_data = unlabel_data_q

        record_all_unlabel_data, predictions_roih_unsup_q, _, _ = self.model(
            all_unlabel_data, branch="supervised_target"
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
                if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                elif key == "loss_bpc_pseudo":
                    loss_dict[key] = record_dict[key] * 0
                elif key[-6:] == "pseudo":  # unsupervised loss
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                elif (key == "loss_DC_img_s" or key == "loss_DC_img_t") and self.cfg.DOMAIN_CLASSIFIER.IMAGE:
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT
                elif (key == "loss_DC_ins_s" or key == "loss_DC_ins_t") and self.cfg.DOMAIN_CLASSIFIER.INSTANCE:
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT
                elif (key == "loss_bpc_pseudo"):
                    loss_dict[key] = record_dict[key] * 0
                else:  # supervised loss
                    loss_dict[key] = record_dict[key] * 0
        self.storage.put_scalar(
            "calibration/bpc_loss", loss_dict["loss_bpc_pseudo"]
        )
        losses = sum(loss_dict.values())

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

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
                        student_model_dict[key] *
                        (1 - keep_rate) + value * keep_rate
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
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                                  test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                                  test_and_save_results_teacher))
        if cfg.TEST.VAL_LOSS:
            ret.append(ValLossHook(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=DatasetMapper(self.cfg, is_train=True)),
                model_name="_student"
            ))
            ret.append(ValLossHook(
                cfg.TEST.EVAL_PERIOD,
                self.model_teacher,
                build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=DatasetMapper(self.cfg, is_train=True))
            ))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret