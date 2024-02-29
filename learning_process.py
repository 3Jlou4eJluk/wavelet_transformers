import numpy as np 
import torch
from tqdm import tqdm
import pickle
import time
from functools import partial
from torch import nn
import pandas as pd


def xavier_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            

class ModelCheckpoint:
    def __init__(self, filepath=None):
        self.val_acc = None
        self.val_loss = None
        self.train_acc = None
        self.train_loss = None
        self.filepath = filepath
        if filepath is None:
            print('<Model Checkpoint>: Warning, model save path is not specified. Setting current directory.')
            self.filepath = '.'

    def update(self, model, train_acc, train_loss, val_acc, val_loss):
        if (self.val_acc is None) or (val_acc > self.val_acc):
            self.val_acc = val_acc
            self.val_loss = val_loss
            self.train_acc = train_acc
            self.train_loss = train_loss
            self.save(model)


    def save(self, model):
        torch.save(model, f"{self.filepath}/model_save.pt")
        with open(f'{self.filepath}/model_checkpoint.pkl', 'wb') as f:
            pickle.dump(self, f)

    def load(self):
        with open(f'{self.filepath}/model_checkpoint.pkl', 'rb') as f:
            loaded_object = pickle.load(f)
            self.val_acc = loaded_object.val_acc
            self.val_loss = loaded_object.val_loss
            self.train_acc = loaded_object.train_acc
            self.train_loss = loaded_object.train_loss
        return torch.load(f"{self.filepath}/model_save.pt")

    def __repr__(self):
        return f"<Model Checkpoint>\n"\
             + f"Validation Accuracy: {self.val_acc}\n"\
             + f"Validation Loss: {self.val_loss}\n"\
             + f"Train Accuracy: {self.train_acc}\n"\
             + f"Train Loss: {self.train_loss}\n"


class Actor:
    def __init__(self, thres=420, sleep_time=120):
        self.prev_time = None
        self.thres = thres
        self.sleep_time = sleep_time

    def action(self, ):
        if self.prev_time is None:
            self.prev_time = time.time()

        if time.time() - self.prev_time > self.thres:
            self.thres = 0
            time.sleep(self.sleep_time)

class Logger:
    def __init__(self, epochs, steps_per_epoch, checkpoints_module=None):
        self.data_df = pd.DataFrame(
            None, columns=['comment', 'step_no', 'loss', 'acc']
        )
        self.last_global_step = None
        self.checkpoints = checkpoints_module

    def log_data(self, data):
        print(f'Logger: {data}')
        for key in data.keys():
            data[key] = [data[key], ]
        self.data_df = pd.concat((self.data_df, pd.DataFrame.from_dict(data)), ignore_index=True)


class Learner:
    def __init__(
            self, model, optimizer, loss_fn, scheduler,
            train_dataloader, val_dataloader, device, epochs,
            actor=None, logger=None, log_period=0.5
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.log_period = log_period
        self.epochs = epochs
        self.device = device


        # service fields
        self.actor = actor
        self.logger = logger

        self.prev_log_step = 0


    def train_epoch(self, log_train_quality=False, verbose=False, epoch_no=0):
        self.model.train()

        loss_sum = []
        acc_sum = []
        for batch_no, (x, y) in enumerate(self.train_dataloader):
            if 'cpu' not in self.device:
                x = x
                y = y.to(self.device)


            self.optimizer.zero_grad()
            pred = self.model.forward(x)
            loss = self.loss_fn(pred, y.squeeze())
            loss.backward()

            pred_classes = torch.argmax(pred, dim=1)
            loss_sum.append(float(loss.item()))
            acc_sum.append(float((
                pred_classes.detach().squeeze() == y.squeeze()
            ).sum() / len(y)))

            global_step = epoch_no * len(self.train_dataloader) + batch_no
            if (self.logger is not None) \
                and (global_step >= int(len(self.train_dataloader) * self.log_period) + self.prev_log_step):

                self.prev_log_step = global_step
                self.logger.log_data(
                    {
                        'comment'       : 'training_process',
                        'step_no'       : global_step,
                        'loss'          : sum(loss_sum) / len(loss_sum),
                        'acc'           : sum(acc_sum) / len(acc_sum)
                    }
                )

            self.optimizer.first_step(zero_grad=True)
            self.loss_fn(self.model.forward(x), y.squeeze())
            self.optimizer.second_step(zero_grad=True)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.actor is not None:
                self.actor.action()

        loss_sum_val = sum(loss_sum) / len(self.train_dl)
        acc_sum_val = sum(acc_sum) / len(self.train_dl)


    def validation_epoch(self, log_train_quality=False, verbose=False, epoch_no=None):
        self.model.eval()

        train_loss = 0.
        val_loss = 0.
        train_acc = 0.
        val_acc = 0.
        with torch.no_grad():
            if log_train_quality:
                for x, y in self.train_dataloader:
                    if 'cpu' not in self.device:
                        x = x
                        y = y.to(self.device)
                    pred = self.model.forward(x)
                    pred_classes = torch.argmax(pred, dim=1)
                    train_acc += float((
                        pred_classes.detach().squeeze() == y.squeeze()
                    ).sum() / len(y))

                    loss = self.loss_fn(pred, y.squeeze())
                    train_loss += float(loss.item())
                train_loss /= len(self.train_dl)
                train_acc /= len(self.train_dl)

            for x, y in tqdm(self.val_dataloader, disable=not verbose):
                if 'cpu' not in self.device:
                    x = x
                    y = y.to(self.device)
                pred = self.model.forward(x)
                pred_classes = torch.argmax(pred, dim=1)
                val_acc += float((
                    pred_classes.detach().squeeze() == y.squeeze()
                ).sum() / len(y))

                loss = self.loss_fn(pred, y.squeeze())
                val_loss += float(loss.item())

            val_acc /= len(self.val_dl)
            val_loss /= len(self.val_dl)

        if self.logger is None:
            return None

        if epoch_no is not None:
            if log_train_quality:
                self.logger.log_data(
                    {
                        'comment' : 'validation_train',
                        'step_no' : epoch_no,
                        'loss'    : train_loss,
                        'acc'     : train_acc
                    }
                )

            self.logger.log_data(
                {
                    'comment' : 'validation_test',
                    'step_no' : epoch_no,
                    'loss'    : val_loss,
                    'acc'     : val_acc
                }
            )


    def train_cycle(self):
        for epoch in (pbar := tqdm(range(self.epochs), total=self.epochs, disable=False)):
            self.train_epoch(log_train_quality=True, verbose=True, epoch_no=epoch)
            if not epoch % 1:
                self.validation_epoch(log_train_quality=False, verbose=True, epoch_no=epoch)


    def train(self):
        if 'cpu' not in self.device:
            self.model.to(self.device)
        self.train_cycle()