#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import streamlit as st
import numpy as np
import cv2

from data import QDataset
from model import QModelSelector
from predictor import QPredictor

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision.ops import box_iou

from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


def get_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])  # x left
    yA = max(boxA[1], boxB[1])  # y top
    xB = min(boxA[2], boxB[2])  # x rigth
    yB = min(boxA[3], boxB[3])  # y bottom
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return float(iou)


@torch.inference_mode()
def evaluate(dataset):

    thresh = 0.1
    
    len_of_data = len(dataset)

    iou_thresh = 0.2

    for i in range(10):
        n_target = 0
        n_pred = 0
        tp = 0
        fp = 0
        fn = 0
        actual_p = 0
        pred_p = 0

        model = QModelSelector().type1(
            box_score_thresh=thresh,
            box_nms_thresh=0.05,
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

        predictor = QPredictor(model)

        for i in range(len_of_data):
            _tp = 0

            x, target = test_dataset[i]
            y_pred = predictor.pred(x.unsqueeze(0).cuda())

            y_pred = {k: v.to("cpu") for k, v in y_pred[0].items()}

            #out = confusion_matrix(target['boxes'], y_pred['boxes'])

            correct_per_image = 0

            for tb in target['boxes']:
                for pb in y_pred['boxes']:
                    iou = get_iou(tb, pb)

                    #print(iou)
                    if iou > iou_thresh:
                        correct_per_image += 1

            _tp = correct_per_image
            _len_target = len(target['boxes'])
            _len_pred = len(y_pred['boxes'])

            tp += _tp
            actual_p += _len_target
            pred_p += _len_pred

            #print(tp, pred_p, actual_p)
        try:
            precision = tp / pred_p
            recall = tp / actual_p
        except Exception as e:
            print(e)
            print(i, tp, pred_p, actual_p)

        print(f'[{thresh}] prec. {precision}, rec. {recall}')


            #y_tensor = torch.tensor(y_pred['boxes']).clone().detach()
            #target_tensor = torch.tensor(target['boxes']).clone().detach()

            #ious = box_iou(
            #    y_tensor,
            #    target_tensor,
            #)
            #print('ious', ious)
            #cm = confusion_matrix = evaluate_single(y, target)

            #x = image_convert(x)
            #x = np.ascontiguousarray(x)
            #
            #n_target += len(target['boxes'])
            #n_pred += len(y['boxes'])

            #tp += cm['tp']
            #fp += cm['fp']
            #fn += cm['fn']

        #precision = tp / (tp + fp)
        #recall = tp / fp + fn

        thresh += 0.1


def evaluate2(dataset):
    coco = get_coco_api_from_dataset(dataset)
    coco_evaluator = CocoEvaluator(coco, ["bbox"])

    len_dataset = len(dataset)

    print(test_dataset)
    #for i in range(len_dataset):
    for i in range(len_dataset):
        x, target = dataset[i]
        y_pred = predictor.pred(x.unsqueeze(0).cuda())
        y_pred = {k: v.to("cpu") for k, v in y_pred[0].items()}

        res = {target["image_id"].item(): y_pred}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator
        
    # evaluator_time = time.time()
    # coco_evaluator.update(res)
    # evaluator_time = time.time() - evaluator_time
    # metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)


from rafael_evaluator.BoundingBox import BoundingBox
from rafael_evaluator.BoundingBoxes import BoundingBoxes
from rafael_evaluator.utils import *
from rafael_evaluator.Evaluator import *

def evaluate3(dataset):
    len_dataset = len(dataset)

    all_bounding_boxes = BoundingBoxes()
    # inferences
    # boxing
    for i in range(len_dataset):
    #for i in range(20):
        x_img, y_gt = dataset[i]
        y_pred = predictor.pred(x_img.unsqueeze(0).cuda())
        #print(y_pred)
        #y_pred = {k: v.to("cpu") for k, v in y_pred[0].items()}
        y_pred = {k: v.cpu().detach().numpy() for k, v in y_pred[0].items()}
        #print(y_pred)
        #torch.cuda.empty_cache()

        for bb in y_gt['boxes']:
            gt_box = BoundingBox(
                imageName=y_gt['image_id'].item(),
                classId='caries',
                x=bb[0],
                y=bb[1],
                w=bb[2],
                h=bb[3],
                typeCoordinates=CoordinatesType.Absolute,
                bbType=BBType.GroundTruth,
                format=BBFormat.XYX2Y2,
                imgSize=tuple(x_img.shape[1:]),
            )
            all_bounding_boxes.addBoundingBox(gt_box)

        for bb, score in zip(y_pred['boxes'], y_pred['scores']):
            pred_box = BoundingBox(
                imageName=y_gt['image_id'].item(),
                classId='caries',
                classConfidence=score.item(),
                x=bb[0],
                y=bb[1],
                w=bb[2],
                h=bb[3],
                typeCoordinates=CoordinatesType.Absolute,
                bbType=BBType.Detected,
                format=BBFormat.XYX2Y2,
                imgSize=(x_img.shape[2], (x_img.shape[1])),
            )
            all_bounding_boxes.addBoundingBox(pred_box)

        _img = image_convert(x_img)
        _img = np.ascontiguousarray(_img)
        #print(_img.shape)
        im = all_bounding_boxes.drawAllBoundingBoxes(_img, y_gt['image_id'].item())
        st.image(im)

    evaluator = Evaluator()

    evaluator.PlotPrecisionRecallCurve(
        all_bounding_boxes,  # Object containing all bounding boxes (ground truths and detections)
        #IOUThreshold=0.3,  # IOU threshold
        IOUThreshold=0.05,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,  # As the official matlab code
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=True,
        savePath='results',
        showGraphic=False,
    )  # Plot the interpolated precision curve

    st.image('results/caries.png')
    #st.pyplot(fig)

    # Get metrics with PASCAL VOC metrics
    metricsPerClass = evaluator.GetPascalVOCMetrics(
        all_bounding_boxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=0.05,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
    print("Average precision values per class:\n")
    # Loop through classes to obtain their metrics
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        # Print AP per class
        print('%s: %f' % (c, average_precision))
        #print(precision, recall, ipre, irec)



    # evaluate

def image_convert(img):
    img = img.clone().cpu().numpy()
    img = img.transpose((1,2,0))
    img = (img * 255).astype(np.uint8)

    return img


st.title('Experimental Results')

test_dataset = QDataset('../data_anony', train = False)
test_dataset.transforms = A.Compose(
    [
        ToTensorV2()
    ],
    bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']},
)

filepath = 'ckpt/bestmodel.pt'
#filepath = 'ckpt/chkpoint_'
checkpoint = torch.load(filepath)

model = QModelSelector().type1(
    box_score_thresh=0.2,
    box_nms_thresh=0.005,
)
model.load_state_dict(checkpoint['state_dict'])
model.cuda()

predictor = QPredictor(model)

#for i in range(5):
#    x, target = test_dataset[i]
#
#    #model.iou = 0.2
#    
#    predictor = QPredictor(model)
#    y = predictor.pred(x.unsqueeze(0).cuda())
#    
#    x = image_convert(x)
#    x = np.ascontiguousarray(x)
#    
#    boxes_pred = y[0]['boxes']
#    boxes_target = target['boxes']
#    for b in boxes_pred:
#        cv2.rectangle(x, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 0), thickness=2 )
#
#    for b in boxes_target:
#        cv2.rectangle(x, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), thickness=2 )
#    
#    #cv2.putText(x, "abc", (x.shape[1]-100, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 0, 0), 2)
#    st.image(x)
#    torch.cuda.empty_cache()

#evaluate(test_dataset)
#evaluate2(test_dataset)
evaluate3(test_dataset)

# Get Average Precision (AP, mAP) from 2022-04-01 14:33
#   ex. case #1
# 
#     target = 4
#     pred   = 6
# 
#     TP = 3        1. checking this
#     FP = 3        2. pred - TP
#     FN = 1        3. target - TP
#     TN = ???
# 
#     precision = TP / (TP + FP)
#     recall = TP / TP + FN


