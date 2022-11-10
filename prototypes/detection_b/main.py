#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import time
import shutil
from skimage import io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary as summary_

import torchvision
from torchvision import transforms
from torchvision import utils

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

#from albumentations.pytorch import ToTensor
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip,
    ShiftScaleRotate,
    VerticalFlip,
    Normalize,
    Flip,
    Compose,
    GaussNoise,
    Resize
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
print(device)

csv_path = 'data/caries.csv'
train_dir = 'data/camera'

df = pd.read_csv(csv_path)
print(df.head())

print()
print(f'Total number of train images is {len(os.listdir(train_dir))}')
print(f'shape of dataframe is {df.shape}')
print(f'Number of images in dataframe is {len(np.unique(df["image_id"]))}')
print(f'Number of train images with no bounding boxes {len(os.listdir(train_dir)) - len(np.unique(df["image_id"]))}')

# splitting to train and validation data
image_ids = df['image_id'].unique()
train_ids = image_ids[0:int(0.8*len(image_ids))]
val_ids = image_ids[int(0.8*len(image_ids)):]
print()
print(f'Total images {len(image_ids)}')
print(f'No of train images {len(train_ids)}')
print(f'No of validation images {len(val_ids)}')

train_df = df[df['image_id'].isin(train_ids)]
val_df = df[df['image_id'].isin(val_ids)]


# helper function for augmentation
def get_transforms(phase):
    list_transforms = []
    
    if phase == 'train':
        list_transforms.extend(
            [
               Flip(p=0.5),
            ]
        )

    list_transforms.extend(
        [
            # ToTensor(),
            # transforms.ToTensor()
            Resize(80, 80),
            ToTensorV2(),
        ]
    )
    
    return Compose(
        list_transforms,
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )

# dataset loader function
class Qset(Dataset):
    def __init__(self, data_frame, image_dir, phase='train'):
        super().__init__()
        self.df = data_frame
        self.image_dir = image_dir
        self.images = data_frame['image_id'].unique()
        self.transforms = get_transforms(phase)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx] + '.jpg'
        target = None
        image_id = None

        image_arr = cv2.imread(os.path.join(self.image_dir, image), cv2.IMREAD_COLOR)
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_arr /= 255.0

        image_id = str(image.split('.')[0])
        point = self.df[self.df['image_id'] == image_id]

        boxes = point[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((point.shape[0],), dtype=torch.int64)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((point.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor(idx)
        target['area'] = area
        target['iscrowd'] = iscrowd

        sample = {
            'image': image_arr,
            'bboxes': target['boxes'],
            'labels': target['labels']
        }

        if self.transforms:
            sample = self.transforms(**sample)

        image = sample['image']
            
        target['boxes'] = torch.stack(
            tuple(map(torch.tensor, zip(*sample['bboxes'])))
        ).permute(1, 0)

        return image, target, image_id

train_data = Qset(train_df, train_dir, phase='train')
val_data = Qset(val_df, train_dir, phase='validataion')

print()
print(f'Length of train data {len(train_data)}')
print(f'Length of validation data {len(val_data)}')

# batching
def collate_fn(batch):
    return tuple(zip(*batch))

train_data_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    val_data,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

# Helper functions for image convertion and visualization
def image_convert(image):
    image = image.clone().cpu().numpy()
    image = image.transpose((1,2,0))
    image = (image * 255).astype(np.uint8)
    return image


def plot_img(data, idx):
    out = data.__getitem__(idx)
    #out = data[idx]
    image = image_convert(out[0])
    image = np.ascontiguousarray(image)
    bb = out[1]['boxes'].numpy()
    for i in bb:
        cv2.rectangle(image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), (0,255,0), thickness=2)



plot_img(train_data, 0)
plot_img(train_data, 22)
plot_img(train_data, 101)
plt.show()

# loading the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
summary_(model, (3, 28, 28), batch_size=8)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

images, targets, ids = next(iter(train_data_loader))
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# optimizer = torch.optim.Adam(params, lr=0.001)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# training
# helper functions to save best model

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


num_epochs = 5
train_loss_min = 0.9
total_train_loss = []

checkpoint_path = 'chkpoint_'
best_model_path = 'bestmodel_may12.pt'

for epoch in range(num_epochs):
    print(f'Epoch :{epoch + 1}')
    start_time = time.time()
    train_loss = []
    model.train()
    for images, targets, image_ids in train_data_loader:
        images = list(image.to(device) for image in images)
        #print(images[0].shape)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        train_loss.append(losses.item())        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # #train_loss/len(train_data_loader.dataset)
    # epoch_train_loss = np.mean(train_loss)
    # total_train_loss.append(epoch_train_loss)
    # print(f'Epoch train loss is {epoch_train_loss}')
    # 
#   #   if lr_scheduler is not None:
#   #       lr_scheduler.step()
    # 
    # # create checkpoint variable and add important data
    # checkpoint = {
    #         'epoch': epoch + 1,
    #         'train_loss_min': epoch_train_loss,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    # }
    # 
    # # save checkpoint
    # save_ckp(checkpoint, False, checkpoint_path, best_model_path)
    # ## TODO: save the model if validation loss has decreased
    # if epoch_train_loss <= train_loss_min:
    #         print('Train loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(train_loss_min,epoch_train_loss))
    #         # save checkpoint as best model
    #         save_ckp(checkpoint, True, checkpoint_path, best_model_path)
    #         train_loss_min = epoch_train_loss
    # 
    # time_elapsed = time.time() - start_time
    # print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# plt.title('Train Loss')
# plt.plot(total_train_loss)
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()
