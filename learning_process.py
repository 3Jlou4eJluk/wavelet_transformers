import numpy as np 
import torch
import tqdm
from functools import partial


class Preprocessor:
    def __init__(self, objects, labels, *funcs):
        self.objects = objects
        self.labels = labels
        self.funcs = [func for func in funcs]

        self.preprocessed_objects = None
    
    def action(self):
        self.preprocessed_objects = self.objects

        # Preprocessing data
        for f in self.funcs:
            self.preprocessed_objects = f(self.preprocessed_objects)
        return self.preprocessed_objects


class Learner:
    def __init__(self, model, optimizer, loss_fn, scheduler, 
                 train_dl, val_dl, device, epochs
                ):
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_loss_epoch_no": [],
            "val_loss_epoch_no": []
        }

        self.model = model
        self.optimzer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self.train_dl = train_dl
        self.val_dl = val_dl

        self.epochs = epochs
        self.device = device

    def train_epoch(self, log_train_quality=False, verbose=False, epoch_no=None):
        self.model.train()

        loss_sum = 0.
        acc_sum = 0.
        for x, y in tqdm(self.train_dl, disabel=not verbose, leave=True):
            def closure(x, y):
                self.optimzer.zero_grad()
                pred = self.model.forward(x.to(self.device))
                loss = self.loss_fn(pred, y.squeeze().to(self.device))
                loss.backward()

                pred_classes = torch.argmax(pred, dim=1)
                loss_sum += float(loss.item())
                acc_sum += (
                    pred_classes.detach().squeeze() == y.to(self.device).squeeze()
                ).sum() / len(y)
            
            closure = partial(closure, x, y)
            self.optimizer.step(closure)
            if self.scheduler is not None:
                self.scheduler.step()
        loss_sum /= len(self.train_dl)
        acc_sum /= len(self.train_dl)

        if log_train_quality and (epoch_no is not None):
            self.metrics['train_loss_epoch_no'].append(epoch_no)
            self.metrics['train_loss'].append(loss_sum)
            self.metrics['train_acc'].append(acc_sum)


    def validation_epoch(self, log_train_quality=False, verbose=False, epoch_no=None):
        self.model.eval()

        train_loss = 0.
        val_loss = 0.
        train_acc = 0.
        val_acc = 0.
        with torch.no_grad():
            if log_train_quality:
                for x, y in tqdm(self.train_dl, disable=not verbose):
                    pred = self.model.forward(x.to(self.device))
                    pred_classes = torch.argmax(pred, dim=1)
                    train_acc += (
                        pred_classes.detach().squeeze() == y.to(self.device).squeeze()
                    ).sum() / len(y)

                    loss = self.loss_fn(pred, y.squeeze().to(self.device))
                    train_loss += float(loss.item())
                train_loss /= len(self.train_dl)
                train_acc /= len(self.train_dl)
            
            for x, y in tqdm(self.val_dl, disable=not verbose):
                pred = self.model.forward(x.to(self.device))
                pred_classes = torch.argmax(pred, dim=1)
                val_acc += (
                    pred_classes.detach().squeeze() == y.to(self.device).squeeze()
                ).sum() / len(y)
                val_loss += float(loss.item())
            
            val_acc /= len(self.val_dl)
            val_loss /= len(self.val_dl)
        
        if epoch_no is not None:
            self.metrics['train_loss_epoch_no'].append(epoch_no)
            self.metrics['val_loss_epoch_no'].append(epoch_no)
            self.metrics['train_loss'].append(train_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_acc'].append(val_acc)

    def train_cycle(self):
        for epoch in range(self.epochs):
            self.train_epoch(log_train_quality=True, verbose=True, epoch_no=epoch)
            if not epoch % 2:
                self.validation_epoch(log_train_quality=False, verbose=True, epoch_no=epoch)

    def action(self):
        self.train_cycle()


class BruteForcer:
    def __init__(self):
        pass

