#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import cv2
import shutil

import torch
import torchvision

from typing import Optional, Callable, Tuple, Any

import streamlit as st
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

import horovod.torch as hvd

import utils
import math
from pprint import pprint

from data import QDataset
from model import QModelSelector
from trainer import QTrainer
from predictor import QPredictor

import wandb



def init(multi_gpu = False):
    device = None

    if torch.cuda.is_available():
        hvd.init()
        if multi_gpu:
            torch.cuda.set_device(hvd.local_rank())
        else:
            #torch.cuda.set_device(0)
            torch.cuda.set_device(1)
        device = "cuda"
    else:
        device = "cpu"

    return torch.device(device)


# batching
def plot_img(data, idx):

    def image_convert(image):
        image = image.clone().cpu().numpy()
        image = image.transpose((1,2,0))
        image = (image * 255).astype(np.uint8)
        return image


    image, label = data[idx]
    image = image_convert(image)

    image = np.ascontiguousarray(image)
    bb = label['boxes'].numpy()
    for box in bboxes:
        cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,255,0), thickness=2)
    #plt.figure(figsize=(10,10))
    #plt.imshow(image)
    st.image(image)

def show(img, labels, name):

    def image_convert(image):
        image = image.clone().cpu().numpy()
        image = image.transpose((1,2,0))
        image = (image * 255).astype(np.uint8)
        return image

    img = image_convert(img)
    img = np.ascontiguousarray(img)
   
    bboxes = labels['boxes']
    #bboxes = labels[0]['boxes'].numpy()
    for box in bboxes:
        cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,255,0), thickness=2)
    cv2.imwrite(f'results/{name}.png', img)



def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, checkpoint_path)
    # if it is a best model, min validation loss
    if is_best:
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(checkpoint_path, best_model_path)
        
def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


def main():

    wandb.init(
        project="test-project",
        entity="hyeonrae"
    )

    wandb.config = {
        "learning_rate": 0.001,
        "epochs": 15,
        "batch_size": 2
    }

    #init(multi_gpu = True)
    #init(multi_gpu = False)
    if torch.cuda.is_available():
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
     

    # data
    QRAYSET_ROOT= '../data_anony'
    train_dataset = QDataset(QRAYSET_ROOT, train = True)
    test_dataset = QDataset(QRAYSET_ROOT, train = False)
    #valid_dataset = QDataset(QRAYSET_ROOT, train = False)

    transforms = A.Compose(
        [
            #torchvision.transforms.ToTensor(),
            A.RandomSizedBBoxSafeCrop(width=700, height=700, erosion_rate=0.2),
            A.OneOf(
                [
                    A.HorizontalFlip(p=0.7),
                    A.VerticalFlip(p=0.7),
                    A.RandomRotate90(p=0.7),
                ]
            ),
            A.transforms.Blur(p=0.3),
            ToTensorV2(),
        ],
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']},
    )

    train_dataset.transforms = transforms
    test_dataset.transforms = transforms

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=hvd.size(),
        rank=hvd.rank()
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        #batch_size=2,
        batch_size=10,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)),
        sampler=train_sampler,
    )
    
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    # model
    model = QModelSelector().type1()
    #model = QModelSelector().type2()
    model.cuda()
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    trainer = QTrainer(model)
    trainer.train(train_data_loader, 15)

    # # eval
    # x, target = test_dataset[0]
    # x = x.unsqueeze(0) # (C, H, W) -> (1, C, H, W)

    # predictor = QPredictor(model)
    # y = predictor.pred(x.cuda())

    # x, target = test_dataset[0]
    # print(target)
    # print(y)
    # show(x, y[0], 'pred')
    # show(x, target, 'gt')


if __name__ == "__main__":
    main()
