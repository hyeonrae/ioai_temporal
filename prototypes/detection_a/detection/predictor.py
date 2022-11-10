#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

class QPredictor:
    def __init__(self, model):
        self.model = model
        model.eval()

        self.loss_fn = self.get_default_loss_fn()

    def pred(self, x):
        #print(self.model)
        #print(x)
        y = self.model(x)
        return y

    def eval(self, x, target):
        y = self.pred(x)
        loss = self.loss_fn(y, target)
        return y, loss

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def get_default_loss_fn(self):
        def loss_fn():
            return None

        return loss_fn
