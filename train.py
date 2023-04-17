#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/2/26 8:52
# @Author : fhh
# @FileName: train.py
# @Software: PyCharm
import time
import torch
import math
import numpy as np
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, precision_recall_curve,auc
import os
import shutil

class DataTrain:
    def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler
        self.device = device

    def train_step(self, train_iter, epochs=None, model_num=0, early_stop = 10000, threshold = 0.55):
        steps = 1
        best_loss = 100000.
        best_loss_acc = 0.
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'saved_models', 'best.pth')
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0
            alpha = 0.4
            for train_data, train_label in train_iter:
                # print(train_data.shape)
                self.model.train()  # 进入训练模式
                # 使数据与模型在同一设备中
                train_data, train_label = train_data.to(self.device), train_label.to(self.device)
                # 模型预测

                y_hat = self.model(train_data)
                # 计算损失
                # unsqueeze插入一个维度
                loss = self.criterion(y_hat, train_label.float().unsqueeze(1))

                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播损失
                loss.backward()
                # 更新参数
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()
                steps += 1



            end_time = time.time()
            epoch_time = end_time - start_time

            model_predictions, true_labels = predict(self.model, train_iter, device=self.device)
            for i in range(len(model_predictions)):
                if model_predictions[i] < threshold:  # threshold
                    model_predictions[i] = 0
                else:
                    model_predictions[i] = 1
            y_hat = model_predictions
            acc1 = accuracy_score(true_labels, y_hat)

            print(f'Model {model_num+1}|Epoch:{epoch:003} | Time:{epoch_time:.2f}s')
            print(f'Train loss:{total_loss / len(train_iter)}')
            print(f'Train acc:{acc1}')

            train_loss = total_loss / len(train_iter)
            if train_loss < best_loss:
                torch.save(self.model.state_dict(), best_model)
                best_loss = train_loss
                best_loss_acc = acc1
                bestlos_epoch = epoch


            if (best_loss < train_loss) and (epoch - bestlos_epoch >= early_stop):
                break



        self.model.load_state_dict(torch.load(best_model))
        print("best_loss = " + str(best_loss))
        print("best_loss_acc = " + str(best_loss_acc))

    def train_step_val(self, train_iter, val_iter, epochs=None, model_num=0, early_stop = 10000, threshold=0.55):
        print("train with val")
        steps = 1
        bestlos_acc1 = 0.
        best_loss = 100000
        bestlos_epoch = 0
        PATH = os.getcwd()
        latest_model = os.path.join(PATH, 'saved_models', 'latest.pth')
        best_model = os.path.join(PATH, 'saved_models', 'best.pth')
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0
            alpha = 0.4
            for train_data, train_label in train_iter:
                # print(train_data.shape)
                self.model.train()  # 进入训练模式
                # 使数据与模型在同一设备中
                train_data, train_label = train_data.to(self.device), train_label.to(self.device)
                # 模型预测

                y_hat = self.model(train_data)
                # 计算损失
                # unsqueeze插入一个维度
                loss = self.criterion(y_hat, train_label.float().unsqueeze(1))

                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播损失
                loss.backward()
                # 更新参数
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()
                steps += 1

            # self.model.eval()
            model_predictions, true_labels = predict(self.model, val_iter, device=self.device)

            for i in range(len(model_predictions)):
                if model_predictions[i] < threshold:  # threshold
                    model_predictions[i] = 0
                else:
                    model_predictions[i] = 1
            y_hat = model_predictions
            acc1 = accuracy_score(true_labels, y_hat)
            train_loss = total_loss / len(train_iter)

            torch.save(self.model.state_dict(), latest_model)
            # latest
            if train_loss < best_loss:
                shutil.copy(latest_model, best_model)
                bestlos_epoch = epoch
                bestlos_acc1 = acc1
                best_loss = train_loss

            end_time = time.time()
            epoch_time = end_time - start_time

            print(f'Model {model_num+1}|Epoch:{epoch:002} | Time:{epoch_time:.2f}s')
            print(f'Train loss:{total_loss / len(train_iter)}')
            print(f'Train acc:{acc1}')

            if (best_loss < train_loss) and (epoch - bestlos_epoch >= early_stop):
                break

        print("best_loss = {:3.1f}".format(best_loss))
        print("best_loss_acc = {:3.1f}".format(bestlos_acc1))
        self.model.load_state_dict(torch.load(best_model))

def predict(model, data, device="cuda"):
    # 模型预测
    model.to(device)
    model.eval()  # 进入评估模式
    predictions = []
    labels = []

    with torch.no_grad():  # 取消梯度反向传播
        for x, y in data:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            score = model(x)
            label = torch.sigmoid(score)  # 将模型预测值映射至0-1之间
            predictions.extend(label.tolist())
            labels.extend(y.tolist())

    return np.array(predictions), np.array(labels)


def get_linear_schedule_with_warmup(optimizer_, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer_, lr_lambda, last_epoch)


class CosineScheduler:
    # 退化学习率
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch - 1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (epoch - 1 - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr
