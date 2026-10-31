# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
RoI heads of the source-free teacher-student detector.

In addition to the standard Faster R-CNN box head, this file implements the
Prompt-Based Pseudo-Box Adjustment (PPBA) branch of Section 3.3.1. PPBA is a
sub-module of the box head: it predicts only geometric offsets from a
proposal-conditioned prompt and therefore refines pseudo boxes without
touching the classification scores produced by the main head.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from .source_free_fast_rcnn import SourceFreeFastRCNNOutputLayers


class PromptPseudoBoxAdjustment(nn.Module):
    """
    Prompt-Based Pseudo-Box Adjustment (PPBA), Section 3.3.1.

    For the i-th proposal r_i = (x_i, y_i, w_i, h_i) of an image of size (W, H):

        g_i = (x_i/W, y_i/H, w_i/W, h_i/H)                    Eq. (5)
        e_i = phi_pos(g_i)                                    positional embedding
        q_i = phi_ms({F_l}, r_i)                              multi-scale prompt
        v_i = RoIAlign(F_roi, r_i)                            RoI representation
        z_i = [phi_fuse(v_i, e_i); q_i]                        fused prompt

    and the box is refined with

        cx' = cx + dx * w,   cy' = cy + dy * h,
        w'  = w * exp(dw),   h'  = h * exp(dh),                Eq. (6)

    The branch predicts only the offsets (dx, dy, dw, dh); classification
    scores are inherited from the main head. PPBA and ATS therefore act on
    disjoint attributes of a pseudo-label: geometry versus confidence.

    With a C4 ResNet-101 backbone (pyramid channels 256/512/1024/2048) and a
    2048-d RoI representation, the default setting adds 1.70M parameters,
    which matches the 1.7M reported in Section 4.2.
    """

    def __init__(
        self,
        input_shapes: Dict[str, ShapeSpec],
        in_features: List[str],
        roi_dim: int = 2048,
        prompt_channels: int = 84,
        prompt_dim: int = 464,
        pool_size: int = 3,
        sampling_ratio: int = 0,
    ):
        super().__init__()
        self.in_features = list(in_features)
        self.pool_size = pool_size

        # phi_ms: one RoIAlign per pyramid level, followed by a 1x1 projection
        # to a shared channel width so that the levels can be concatenated.
        self.level_poolers = nn.ModuleList(
            [
                ROIPooler(
                    output_size=pool_size,
                    scales=[1.0 / input_shapes[f].stride],
                    sampling_ratio=sampling_ratio,
                    pooler_type="ROIAlignV2",
                )
                for f in self.in_features
            ]
        )
        self.level_proj = nn.ModuleList(
            [
                nn.Conv2d(input_shapes[f].channels, prompt_channels, kernel_size=1)
                for f in self.in_features
            ]
        )
        self.prompt_proj = nn.Linear(len(self.in_features) * prompt_channels, prompt_dim)

        # phi_pos: lightweight MLP on the normalized proposal geometry.
        self.pos_mlp = nn.Sequential(
            nn.Linear(4, prompt_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(prompt_dim // 4, prompt_dim),
        )

        # phi_fuse: channel-wise concatenation of v_i and e_i, then a linear layer.
        self.fuse = nn.Linear(roi_dim + prompt_dim, prompt_dim)

        # Offset head: z_i = [phi_fuse(v_i, e_i); q_i] -> (dx, dy, dw, dh).
        self.offset_head = nn.Linear(2 * prompt_dim, 4)

        # Start from the identity mapping so that refinement is a no-op at the
        # beginning of adaptation and grows with the detector.
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def num_extra_parameters(self) -> int:
        """Number of parameters introduced on top of the base detector."""
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _per_image(feature: torch.Tensor, num_images: int) -> List[torch.Tensor]:
        """Accept either a batched tensor or a list holding one map per image."""
        if isinstance(feature, (list, tuple)):
            return list(feature)
        if feature.dim() == 4:
            return [feature[i] for i in range(num_images)]
        return [feature] * num_images

    def _prompt(self, features: Dict[str, torch.Tensor], index: int, boxes: Boxes,
                num_images: int) -> torch.Tensor:
        """q_i in Eq. (5): the multi-scale prompt of one image."""
        parts = []
        for level, (pooler, proj) in enumerate(zip(self.level_poolers, self.level_proj)):
            feature = self._per_image(features[self.in_features[level]], num_images)[index]
            pooled = pooler([feature], [boxes])
            pooled = proj(pooled)
            pooled = F.adaptive_avg_pool2d(pooled, 1).flatten(1)
            parts.append(pooled)
        return self.prompt_proj(torch.cat(parts, dim=1))

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        roi_features: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Args:
            features: backbone feature maps, keyed by level name.
            proposals: per-image proposals whose boxes are to be refined.
            roi_features: (sum_i N_i, D) RoI representations computed from the
                unrefined proposals by the box pooler and the box head.

        Returns:
            A list with one (N_i, 4) tensor of offsets per image.
        """
        num_images = len(proposals)
        outputs = []
        start = 0
        for index, instances in enumerate(proposals):
            boxes = instances.proposal_boxes
            count = len(boxes)
            if count == 0:
                outputs.append(boxes.tensor.new_zeros((0, 4)))
                start += count
                continue

            height, width = instances.image_size
            x1, y1, x2, y2 = boxes.tensor.unbind(dim=1)
            # g_i in Eq. (5): proposal geometry normalized by the image size.
            geometry = torch.stack(
                [
                    (x1 + x2) / 2.0 / width,
                    (y1 + y2) / 2.0 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ],
                dim=1,
            )
            e_i = self.pos_mlp(geometry)
            q_i = self._prompt(features, index, boxes, num_images)
            v_i = roi_features[start:start + count]
            start += count

            z_i = torch.cat([self.fuse(torch.cat([v_i, e_i], dim=1)), q_i], dim=1)
            outputs.append(self.offset_head(z_i))
        return outputs

    @staticmethod
    def refine_boxes(boxes: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Apply the predicted offsets, Eq. (6). Boxes and offsets are (N, 4)."""
        if boxes.numel() == 0:
            return boxes
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        width = (x2 - x1).clamp(min=1e-3)
        height = (y2 - y1).clamp(min=1e-3)
        center_x = x1 + width / 2.0
        center_y = y1 + height / 2.0

        delta_x, delta_y, delta_w, delta_h = deltas.unbind(dim=1)
        # The refinement stays local: the scale terms are bounded so that a
        # single step cannot collapse or explode a pseudo box.
        delta_w = delta_w.clamp(-1.0, 1.0)
        delta_h = delta_h.clamp(-1.0, 1.0)

        center_x = center_x + delta_x * width
        center_y = center_y + delta_y * height
        width = width * torch.exp(delta_w)
        height = height * torch.exp(delta_h)

        refined = torch.stack(
            [
                center_x - width / 2.0,
                center_y - height / 2.0,
                center_x + width / 2.0,
                center_y + height / 2.0,
            ],
            dim=1,
        )
        return refined.clamp(min=0.0)


@ROI_HEADS_REGISTRY.register()
class SourceFreeAdaptiveTeacherStandardROIHeads(StandardROIHeads):
    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        ppba: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            **kwargs,
        )
        # PPBA lives inside the RoI head so that every pseudo box produced by
        # the teacher is refined before it reaches the filtering step.
        self.ppba = ppba

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = SourceFreeFastRCNNOutputLayers(cfg, box_head.output_shape)
        else:
            raise ValueError("Unknown ROI head loss.")

        ppba = None
        if cfg.MODEL.PPBA.ENABLED:
            # Only the pyramid levels that the backbone actually produces can
            # contribute to the multi-scale prompt phi_ms.
            levels = [f for f in cfg.MODEL.PPBA.IN_FEATURES if f in input_shape]
            if levels:
                ppba = PromptPseudoBoxAdjustment(
                    input_shapes=input_shape,
                    in_features=levels,
                    roi_dim=cfg.MODEL.ROI_BOX_HEAD.FC_DIM,
                    prompt_channels=cfg.MODEL.PPBA.PROMPT_CHANNELS,
                    prompt_dim=cfg.MODEL.PPBA.PROMPT_DIM,
                    pool_size=cfg.MODEL.PPBA.POOL_SIZE,
                )

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "ppba": ppba,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
            losses, _, box_features, instance_proposals = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )
            return proposals, losses, box_features, instance_proposals
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )
            return pred_instances, predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        feature_maps = features
        feature_list = [feature_maps[f] for f in self.box_in_features]
        box_features = self.box_pooler(
            feature_list, [x.proposal_boxes for x in proposals]
        )
        box_features = self.box_head(box_features)

        # PPBA, Eq. (5) and Eq. (6): the prompt is built from the proposal
        # geometry and the multi-scale context, and the predicted offsets
        # replace the proposal coordinates before the box head regresses them.
        # The RoI representation itself is pooled from the unrefined proposals,
        # which keeps the branch conditioned on the teacher's own detections.
        if self.ppba is not None:
            deltas = self.ppba(feature_maps, proposals, box_features)
            for instances, delta in zip(proposals, deltas):
                instances.proposal_boxes = Boxes(
                    self.ppba.refine_boxes(instances.proposal_boxes.tensor, delta)
                )

        predictions = self.box_predictor(box_features)

        if (
            self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss
            losses = self.box_predictor.losses(predictions, proposals)
            with torch.no_grad():
                pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                    predictions, proposals
                )
                for proposals_per_image, pred_boxes_per_image in zip(
                    proposals, pred_boxes
                ):
                    proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            instance_proposals, _ = self.box_predictor.convert_bbox_scores(
                predictions, proposals
            )
            return losses, predictions, box_features, instance_proposals
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt
