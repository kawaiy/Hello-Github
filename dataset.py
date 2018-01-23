#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import numpy as np


class Dataset(object):
    def __init__(self, dataset_name, valid_data_ratio=0.):

        self.dataset_name = dataset_name

        # load dataset
        train, test = None, None
        if dataset_name == 'cifar10':
            self.n_class = 10
            self.channel = 3
            self.pad_size = 4
            train, test = chainer.datasets.get_cifar10(withlabel=True, ndim=3, scale=255.0)
        elif dataset_name == 'cifar100':
            self.n_class = 100
            self.channel = 3
            self.pad_size = 4
            train, test = chainer.datasets.get_cifar100(withlabel=True, ndim=3, scale=255.0)
        elif dataset_name == 'mnist':
            self.n_class = 10
            self.channel = 1
            self.pad_size = 4
            train, test = chainer.datasets.get_mnist(withlabel=True, ndim=2, scale=1.0)
        elif dataset_name == 'mnist_flat':
            self.n_class = 10
            train, test = chainer.datasets.get_mnist(withlabel=True, ndim=1, scale=1.0)
        else:
            print('Invalid dataset name at dataset()!')
            exit(1)

        # split into train and validation data
        np.random.seed(2017)    # for generating fixed validation set
        order = np.random.permutation(len(train))
        np.random.seed()
        train_size = int(len(train) * (1. - valid_data_ratio))

        # train data
        self.x_train, self.y_train = train[order[:train_size]][0], train[order[:train_size]][1]
        # validation data
        self.x_valid, self.y_valid = train[order[train_size:]][0], train[order[train_size:]][1]
        # test data
        self.x_test, self.y_test = test[range(len(test))][0], test[range(len(test))][1]


        # Subtract the mean of training data for each channel and divide by the std
        if dataset_name == 'cifar10' or  dataset_name == 'cifar100':
            mean = self.x_train.mean(axis=(0, 2, 3))[np.newaxis, :, np.newaxis, np.newaxis]
            std = self.x_train.std(axis=(0, 2, 3))[np.newaxis, :, np.newaxis, np.newaxis]
        else:
            mean = self.x_train.mean()
            std = self.x_train.std()
        #print(mean)
        #print(std)

        self.x_train = (self.x_train - mean) / std
        self.x_valid = (self.x_valid - mean) / std
        self.x_test = (self.x_test - mean) / std

        #x_mean = 0.
        #for x in self.x_train:
        #    x_mean += x
        #x_mean /= len(self.x_train)
        #self.x_train -= x_mean
        #self.x_valid -= x_mean
        #self.x_test -= x_mean

    def data_augmentation(self, x_train):
        if self.dataset_name != 'cifar10' and self.dataset_name != 'cifar100':
            print('Our data augmentation does not support this dataset...')
            return x_train

        _, c, h, w = x_train.shape
        pad_h = h + 2 * self.pad_size
        pad_w = w + 2 * self.pad_size
        aug_data = np.zeros_like(x_train)
        for i, x in enumerate(x_train):
            pad_img = np.zeros((c, pad_h, pad_w))
            pad_img[:, self.pad_size:h+self.pad_size, self.pad_size:w+self.pad_size] = x

            # Randomly crop and horizontal flip the image
            top = np.random.randint(0, pad_h - h + 1)
            left = np.random.randint(0, pad_w - w + 1)
            bottom = top + h
            right = left + w
            aug_data[i] = pad_img[:, top:bottom, left:right]

            if np.random.randint(0, 2):
                aug_data[i] = aug_data[i][:, :, ::-1]

        return aug_data