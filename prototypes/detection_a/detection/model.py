#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import torch
import torchvision

#class QModel(torch.nn.Module):
class QModelSelector():
    def __init__(self):
        self.num_classes = 2

    def type1(
        self,
        box_score_thresh=0.2,
        box_nms_thresh=0.005,
    ):
        num_classes = self.num_classes

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            box_score_thresh=box_score_thresh, # default 0.05
            box_nms_thresh=box_nms_thresh, # default 0.5
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            num_classes
        )
        return model

    def type2(self):
        # model modifying/
        num_classes = self.num_classes

        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        #backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        backbone.out_channels = 1280

        anchor_sizes = ((32, 64, 128, 256, 512),)
        #aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        aspect_ratios = ((0.5, 1.0, 2.0),)
        anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2,
        )

        model = torchvision.models.detection.FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            box_score_thresh=0.05, # default 0.05
            box_nms_thresh=0.1, # default 0.5
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.2,
        )
        return model

if __name__ == "__main__":
    # test
    model = QModel()
    print(model)

    model = QModel().type2()
    print(model)
