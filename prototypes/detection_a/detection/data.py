#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Optional, Callable, Tuple, Any

import numpy as np
import cv2
import torch
import pandas as pd
import albumentations as A

import zlib
import base64

# data -> bitmap
def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    # n = np.fromstring(z, np.uint8)
    n = np.frombuffer(z, np.uint8)
    # mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(np.uint8)
    return mask


def attention_mask(img, b64, origin_x, origin_y):
    imh, imw, _ = img.shape

    img_mask = np.zeros([imh, imw], dtype=np.uint8)
    img_zero = np.zeros([imh, imw], dtype=np.uint8)

    for m, orix, oriy in zip(b64, origin_x, origin_y):
        data_b64 = m

        local_mask = base64_2_mask(data_b64)
        local_shape = local_mask.shape

        img_mask[
            oriy : oriy + local_shape[0], orix : orix + local_shape[1]
        ] += local_mask

    color_mask = cv2.merge((img_mask, img_mask, img_mask))
    blurred_mask = cv2.GaussianBlur(color_mask, (51, 51), 0.0)

    return cv2.subtract(img, cv2.bitwise_not(blurred_mask))



class QDataset(torch.utils.data.Dataset):
    """ Camera-Qray Dental Dataset.

    """

    resources = {
        "dirs": {
            "camera": "camera",
            "qray": "qray",
        },
        "anns": {
            "caries": "caries.csv",
            "global": "global.csv",
            "tooth": "tooth.csv",
        },
    }

    training_file = "train.samples"
    test_file = "test.samples"

    classes = {
        "global": [
            "nomal",
            "anomal",
        ],
    }

    _image_exc = ".jpg"

    def __init__(
            self,
            root: str,
            target_type: Optional = None,
            train: bool = True,
            transforms: Optional[Callable] = None
    ) -> None:

        # super().__init__()

        self.root = root
        self.target_type = target_type
        self.train = train
        self._transforms = transforms

        self.anns = {
            "global": self._load_csv(self.resources["anns"]["global"]),
            "caries": self._load_csv(self.resources["anns"]["caries"]),
            "tooth": self._load_csv(self.resources["anns"]["tooth"]),
        }

        self.X, self.Y = self._load_data()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        x_id = self.X[idx]
        x_filename = x_id + self._image_exc
        x_filepath = self._getpath(os.path.join(self.resources['dirs']['camera'], x_filename))

        y_ = None

        x_image = cv2.imread(x_filepath, cv2.IMREAD_COLOR)

        # image preprocessing
        if self.train:
            #try:
            #    cur_tooth = self.anns['tooth'][self.anns['tooth']['image_id'] == x_id]
            #    _mask = cur_tooth['mask'].values[0]
            #    _orix = cur_tooth['origin_x'].values[0]
            #    _oriy = cur_tooth['origin_y'].values[0]
            #    x_image = attention_mask(x_image, _mask, (_orix, _oriy))
            #except:
            #    pass

            cur_tooth = self.anns['tooth'][self.anns['tooth']['image_id'] == x_id]
            _mask = cur_tooth['mask'].values
            _orix = cur_tooth['origin_x'].values
            _oriy = cur_tooth['origin_y'].values
            x_image = attention_mask(x_image, _mask, _orix, _oriy)
            pass
        else:
            cur_tooth = self.anns['tooth'][self.anns['tooth']['image_id'] == x_id]
            _mask = cur_tooth['mask'].values
            _orix = cur_tooth['origin_x'].values
            _oriy = cur_tooth['origin_y'].values
            x_image = attention_mask(x_image, _mask, _orix, _oriy)
        # /image preprocessing
        
        x_image = cv2.cvtColor(x_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        x_image /= 255.0

        #st.image(x_image)

        point = self.anns['caries'][self.anns['caries']['image_id'] == x_id]

        # x, y, w, h -> tlx, tly, brx, bry, tl: topleft, br: bottom right
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

        #print(target)

        sample = {
            'image': x_image,
            'bboxes': target['boxes'],
            'labels': target['labels']
        }

        if self.transforms:
            sample = self.transforms(**sample)

        x = sample['image']

        #target['boxes'] = torch.FloatTensor(sample['bboxes'])
        target['boxes'] = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
        y = target

        return x, y

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, transforms: Optional[Callable] = None):
        #assert isinstance(transforms, torchvision.transforms.Compose)
        assert isinstance(transforms, A.Compose)
        self._transforms = transforms

    def _getpath(self, name: str = None):
        path = os.path.join(self.root, name)
        return path if os.path.exists(path) else None

    def _load_csv(self, name: str = None) -> pd.DataFrame:
        try:
            return pd.read_csv(self._getpath(name), sep=",")
        except Exception:
            return None

    def _load_sample_file(self, name: str = None):
        try:
            ids = pd.read_csv(self._getpath(name), sep=",", header=None, names=["image_id"])
            return ids["image_id"].unique()
        except Exception:
            return None

    def _load_data(self):
        # Indexes
        #full_image_ids = self.anns['global']['image_id'].unique()
        full_image_ids = sorted(self.anns['caries']['image_id'].unique())

        # TODO Temporarily force data range
        train_range = [0, int(0.8*len(full_image_ids))]
        test_range = [train_range[1], len(full_image_ids)]

        train_ids = self._load_sample_file(self.training_file)
        test_ids = self._load_sample_file(self.test_file)

        is_test = True if test_ids is not None else False
        is_train = True if train_ids is not None else False

        if not is_test and not is_train: 
            test_ids = full_image_ids[test_range[0]:test_range[1]]
            train_ids = full_image_ids[train_range[0]:train_range[1]]
        elif is_test and not is_train: 
            train_ids = np.array([x for x in full_image_ids if x not in test_ids])
        elif not is_test and is_train:
            test_ids = np.array([x for x in full_image_ids if x not in train_ids])

        train_df = self.anns['global'][self.anns['global']['image_id'].isin(train_ids)]
        test_df = self.anns['global'][self.anns['global']['image_id'].isin(test_ids)]

        # Load data
        X_list = []
        Y_list = []

        if self.train:  # for training
            X_list = train_ids
            Y_list = {
                "global": self.anns['global'][self.anns['global']['image_id'].isin(train_ids)],
                "caries": self.anns['caries'][self.anns['caries']['image_id'].isin(train_ids)],
                "tooth": self.anns['tooth'][self.anns['tooth']['image_id'].isin(train_ids)],
            }
        else:  # for testing
            X_list = test_ids
            Y_list = {
                "global": self.anns['global'][self.anns['global']['image_id'].isin(test_ids)],
                "caries": self.anns['caries'][self.anns['caries']['image_id'].isin(test_ids)],
                "tooth": self.anns['tooth'][self.anns['tooth']['image_id'].isin(test_ids)],
            }

        return X_list, Y_list



if __name__ == "__main__":
    pass
