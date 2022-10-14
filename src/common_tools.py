# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.models as models


class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch, num_labels):
        model.train()

        conf_mat = np.zeros((num_labels, num_labels))
        loss_sigma = []

        for i, data in enumerate(data_loader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.int64)

            outputs = model(inputs)

            optimizer.zero_grad()
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            loss_sigma.append(loss.item())
            acc_avg = conf_mat.trace() / conf_mat.sum()

            if i % 50 == 50 - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch_id + 1, max_epoch, i + 1, len(data_loader), np.mean(loss_sigma), acc_avg))

        return np.mean(loss_sigma), acc_avg, conf_mat

    @staticmethod
    def valid(data_loader, model, loss_f, device, num_labels):
        model.eval()

        conf_mat = np.zeros((num_labels, num_labels))
        loss_sigma = []

        with torch.no_grad():
            for i, data in enumerate(data_loader):

                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device, dtype = torch.int64)

                outputs = model(inputs)
                loss = loss_f(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                for j in range(len(labels)):
                    cate_i = labels[j].cpu().numpy()
                    pre_i = predicted[j].cpu().numpy()
                    conf_mat[cate_i, pre_i] += 1.

                loss_sigma.append(loss.item())

        acc_avg = conf_mat.trace() / conf_mat.sum()

        return np.mean(loss_sigma), acc_avg, conf_mat


def show_confMat(confusion_mat, classes, set_name, out_dir, verbose=False):

    cls_num = len(classes)
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    cmap = plt.cm.get_cmap('Greys') #http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)

    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix' + set_name + '.png'))
    # plt.show()
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))))


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    :param train_x: epoch
    :param train_y: 
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()


