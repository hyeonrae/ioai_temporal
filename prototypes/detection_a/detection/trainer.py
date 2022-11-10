#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import torch
import horovod.torch as hvd
import numpy as np

import wandb


class QTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = self.get_optimizer(True)
        self.lr_scheduler = self.get_lr_scheduler(self.optimizer)

        self.checkpoint_path = 'ckpt/chkpoint_'
        self.best_model_path = 'ckpt/bestmodel.pt'

    def train(self, data_loader, num_epochs):
        train_loss_min = 0.9
        total_train_loss = []

        wandb.watch(self.model)
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch+1}')

            self.model.train()
            train_loss = []
            
            for batch_idx, (images, targets) in enumerate(data_loader):
                images = list(image.cuda() for image in images)
                targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)

                #out = model(images)
                #loss_dict = criterion(out, targets)

                losses = sum(loss for loss in loss_dict.values())
                train_loss.append(losses.item())

                if batch_idx % 10 == 9:
                    print(batch_idx, losses.item())

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

            epoch_train_loss = np.mean(train_loss)
            total_train_loss.append(epoch_train_loss)
            print(f'Epoch train loss is {epoch_train_loss}')

            wandb.log(
                {"loss": epoch_train_loss}
            )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            checkpoint = {
                'epoch': epoch + 1,
                'train_loss_min': epoch_train_loss,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

            self.save(checkpoint, self.checkpoint_path, False, self.best_model_path)

            if epoch_train_loss <= train_loss_min:
                print('Train loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(train_loss_min, epoch_train_loss))
                # save checkpoint as best model
                self.save(checkpoint, self.checkpoint_path, True, self.best_model_path)
                train_loss_min = epoch_train_loss

    def get_optimizer(self, multi_gpu=False):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            #model.parameters(),
            lr=0.005,
            #lr=1e-5,
            momentum=0.9,
            weight_decay=0.0005,
        )

        # if is_horovod_available:
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=self.model.named_parameters()
        )

        if multi_gpu:
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        return optimizer

    def get_lr_scheduler(self, optimizer):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )
        return lr_scheduler

    def save(self, state, checkpoint_path=None, is_best=False, best_model_path=None):
        torch.save(state, checkpoint_path)
        if is_best:
            shutil.copyfile(checkpoint_path, best_model_path)


