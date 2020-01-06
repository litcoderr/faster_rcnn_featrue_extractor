from torchvision import models
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import boxes as box_ops
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List, Dict
from collections import OrderedDict
import copy
import torch
import torch.nn as nn


class FasterRcnnFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FasterRcnnFeatureExtractor, self).__init__()
        self.num_classes = 91
        self.device = 'cuda'

        self.transform = original_model.transform
        self.backbone = original_model.backbone
        self.rpn = original_model.rpn

        self.box_roi_pool = original_model.roi_heads.box_roi_pool
        self.box_head = original_model.roi_heads.box_head
        self.box_predictor = original_model.roi_heads.box_predictor
        self.box_coder = original_model.roi_heads.box_coder
        self.detections_per_img = original_model.roi_heads.detections_per_img

        # After done, free original_model
        del original_model

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        # Region Proposal
        proposals, proposal_losses = self.rpn(images, features, targets)
        # ROI Heads
        box_features = self.box_roi_pool(features, proposals, images.image_sizes)
        box_features = self.box_head(box_features)  # Feature Extracted [batch * 1000, 1024]
        class_logits, box_regression = self.box_predictor(box_features)

        # box prediction
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        features_list = box_features.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_features = []
        for boxes, scores, image_shape, box_features in zip(pred_boxes_list, pred_scores_list, images.image_sizes, features_list):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(self.num_classes, device=self.device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, 1.0) # keep all detections
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]

            boxes = boxes/image_shape[0]  # Normalize coord value to 0 to 1
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            box_feature_keep = keep / (self.num_classes-1)  # Select corresponding box feature index
            box_features = box_features[box_feature_keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_features.append(box_features)

        return all_boxes, all_features

    @classmethod
    def build_pretrained(cls):
        f_rcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().cuda()
        custom = cls(f_rcnn).eval().cuda()
        return custom
