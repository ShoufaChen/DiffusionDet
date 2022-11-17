# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Rufeng Zhang, Peize Sun
# Contact: {sunpeize, cxrfzhang}@foxmail.com
# 
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
from itertools import count
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.modeling import GeneralizedRCNNWithTTA, DatasetMapperTTA
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference_single_image
from detectron2.structures import Instances, Boxes


class DiffusionDetWithTTA(GeneralizedRCNNWithTTA):
    """
        A DiffusionDet with test-time augmentation enabled.
        Its :meth:`__call__` method has the same interface as :meth:`DiffusionDet.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
        """
            Args:
                cfg (CfgNode):
                model (DiffusionDet): a DiffusionDet to apply TTA on.
                tta_mapper (callable): takes a dataset dict and returns a list of
                    augmented versions of the dataset dict. Defaults to
                    `DatasetMapperTTA(cfg)`.
                batch_size (int): batch the augmented images into this batch size for inference.
        """
        # fix the issue: cannot assign module before Module.__init__() call
        nn.Module.__init__(self)
        if isinstance(model, DistributedDataParallel):
            model = model.module

        self.cfg = cfg.clone()
        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

        # cvpods tta.
        self.enable_cvpods_tta = cfg.TEST.AUG.CVPODS_TTA
        self.enable_scale_filter = cfg.TEST.AUG.SCALE_FILTER
        self.scale_ranges = cfg.TEST.AUG.SCALE_RANGES
        self.max_detection = cfg.MODEL.DiffusionDet.NUM_PROPOSALS

    def _batch_inference(self, batched_inputs, detected_instances=None):
        """
        Execute inference on a list of inputs,
        using batch size = self.batch_size, instead of the length of the list.

        Inputs & outputs have the same format as :meth:`DiffusionDet.forward`
        """
        if detected_instances is None:
            detected_instances = [None] * len(batched_inputs)

        factors = 2 if self.tta_mapper.flip else 1
        if self.enable_scale_filter:
            assert len(batched_inputs) == len(self.scale_ranges) * factors

        outputs = []
        inputs, instances = [], []
        for idx, input, instance in zip(count(), batched_inputs, detected_instances):
            inputs.append(input)
            instances.append(instance)
            if self.enable_cvpods_tta:
                output = self.model.forward(inputs, do_postprocess=False)[0]
                if self.enable_scale_filter:
                    pred_boxes = output.get("pred_boxes")
                    keep = self.filter_boxes(pred_boxes.tensor, *self.scale_ranges[idx // factors])
                    output = Instances(
                        image_size=output.image_size,
                        pred_boxes=Boxes(pred_boxes.tensor[keep]),
                        pred_classes=output.pred_classes[keep],
                        scores=output.scores[keep])
                outputs.extend([output])
            else:

                if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
                    outputs.extend(
                        self.model.forward(
                            inputs,
                            do_postprocess=False,
                        )
                    )
            inputs, instances = [], []
        return outputs

    @staticmethod
    def filter_boxes(boxes, min_scale, max_scale):
        """
        boxes: (N, 4) shape
        """
        # assert boxes.mode == "xyxy"
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
        return keep

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor

        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)
        # Detect boxes from all augmented versions
        all_boxes, all_scores, all_classes = self._get_augmented_boxes(augmented_inputs, tfms)
        # merge all detected boxes to obtain final predictions for boxes
        if self.enable_cvpods_tta:
            merged_instances = self._merge_detections_cvpods_tta(all_boxes, all_scores, all_classes, orig_shape)
        else:
            merged_instances = self._merge_detections(all_boxes, all_scores, all_classes, orig_shape)

        return {"instances": merged_instances}

    def _merge_detections(self, all_boxes, all_scores, all_classes, shape_hw):
        # select from the union of all results
        num_boxes = len(all_boxes)
        num_classes = self.cfg.MODEL.DiffusionDet.NUM_CLASSES
        # +1 because fast_rcnn_inference expects background scores as well
        all_scores_2d = torch.zeros(num_boxes, num_classes + 1, device=all_boxes.device)
        for idx, cls, score in zip(count(), all_classes, all_scores):
            all_scores_2d[idx, cls] = score

        merged_instances, _ = fast_rcnn_inference_single_image(
            all_boxes,
            all_scores_2d,
            shape_hw,
            1e-8,
            self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            self.cfg.TEST.DETECTIONS_PER_IMAGE,
        )

        return merged_instances

    def _merge_detections_cvpods_tta(self, all_boxes, all_scores, all_classes, shape_hw):
        all_scores = torch.tensor(all_scores).to(all_boxes.device)
        all_classes = torch.tensor(all_classes).to(all_boxes.device)

        all_boxes, all_scores, all_classes = self.merge_result_from_multi_scales(
            all_boxes, all_scores, all_classes,
            nms_type="soft_vote", vote_thresh=0.65,
            max_detection=self.max_detection
        )

        all_boxes = Boxes(all_boxes)
        all_boxes.clip(shape_hw)

        result = Instances(shape_hw)
        result.pred_boxes = all_boxes
        result.scores = all_scores
        result.pred_classes = all_classes.long()
        return result

    def merge_result_from_multi_scales(
            self, boxes, scores, labels, nms_type="soft-vote", vote_thresh=0.65, max_detection=100
    ):
        boxes, scores, labels = self.batched_vote_nms(
            boxes, scores, labels, nms_type, vote_thresh
        )

        number_of_detections = boxes.shape[0]
        # Limit to max_per_image detections **over all classes**
        if number_of_detections > max_detection > 0:
            boxes = boxes[:max_detection]
            scores = scores[:max_detection]
            labels = labels[:max_detection]

        return boxes, scores, labels

    def batched_vote_nms(self, boxes, scores, labels, vote_type, vote_thresh=0.65):
        # apply per class level nms, add max_coordinates on boxes first, then remove it.
        labels = labels.float()
        max_coordinates = boxes.max() + 1
        offsets = labels.reshape(-1, 1) * max_coordinates
        boxes = boxes + offsets

        boxes, scores, labels = self.bbox_vote(boxes, scores, labels, vote_thresh, vote_type)
        boxes -= labels.reshape(-1, 1) * max_coordinates

        return boxes, scores, labels

    def bbox_vote(self, boxes, scores, labels, vote_thresh, vote_type="softvote"):
        assert boxes.shape[0] == scores.shape[0] == labels.shape[0]
        det = torch.cat((boxes, scores.reshape(-1, 1), labels.reshape(-1, 1)), dim=1)

        vote_results = torch.zeros(0, 6, device=det.device)
        if det.numel() == 0:
            return vote_results[:, :4], vote_results[:, 4], vote_results[:, 5]

        order = scores.argsort(descending=True)
        det = det[order]

        while det.shape[0] > 0:
            # IOU
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            xx1 = torch.max(det[0, 0], det[:, 0])
            yy1 = torch.max(det[0, 1], det[:, 1])
            xx2 = torch.min(det[0, 2], det[:, 2])
            yy2 = torch.min(det[0, 3], det[:, 3])
            w = torch.clamp(xx2 - xx1, min=0.)
            h = torch.clamp(yy2 - yy1, min=0.)
            inter = w * h
            iou = inter / (area[0] + area[:] - inter)

            # get needed merge det and delete these det
            merge_index = torch.where(iou >= vote_thresh)[0]
            vote_det = det[merge_index, :]
            det = det[iou < vote_thresh]

            if merge_index.shape[0] <= 1:
                vote_results = torch.cat((vote_results, vote_det), dim=0)
            else:
                if vote_type == "soft_vote":
                    vote_det_iou = iou[merge_index]
                    det_accu_sum = self.get_soft_dets_sum(vote_det, vote_det_iou)
                elif vote_type == "vote":
                    det_accu_sum = self.get_dets_sum(vote_det)
                vote_results = torch.cat((vote_results, det_accu_sum), dim=0)

        order = vote_results[:, 4].argsort(descending=True)
        vote_results = vote_results[order, :]

        return vote_results[:, :4], vote_results[:, 4], vote_results[:, 5]

    @staticmethod
    def get_dets_sum(vote_det):
        vote_det[:, :4] *= vote_det[:, 4:5].repeat(1, 4)
        max_score = vote_det[:, 4].max()
        det_accu_sum = torch.zeros((1, 6), device=vote_det.device)
        det_accu_sum[:, :4] = torch.sum(vote_det[:, :4], dim=0) / torch.sum(vote_det[:, 4])
        det_accu_sum[:, 4] = max_score
        det_accu_sum[:, 5] = vote_det[0, 5]
        return det_accu_sum

    @staticmethod
    def get_soft_dets_sum(vote_det, vote_det_iou):
        soft_vote_det = vote_det.detach().clone()
        soft_vote_det[:, 4] *= (1 - vote_det_iou)

        INFERENCE_TH = 0.05
        soft_index = torch.where(soft_vote_det[:, 4] >= INFERENCE_TH)[0]
        soft_vote_det = soft_vote_det[soft_index, :]

        vote_det[:, :4] *= vote_det[:, 4:5].repeat(1, 4)
        max_score = vote_det[:, 4].max()
        det_accu_sum = torch.zeros((1, 6), device=vote_det.device)
        det_accu_sum[:, :4] = torch.sum(vote_det[:, :4], dim=0) / torch.sum(vote_det[:, 4])
        det_accu_sum[:, 4] = max_score
        det_accu_sum[:, 5] = vote_det[0, 5]

        if soft_vote_det.shape[0] > 0:
            det_accu_sum = torch.cat((det_accu_sum, soft_vote_det), dim=0)
        return det_accu_sum
