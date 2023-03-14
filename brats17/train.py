# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic
# Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
#
from __future__ import absolute_import, print_function

import numpy as np
import random
from scipy import ndimage
import time
import os
import sys
import tensorflow as tf
# from tensorflow.contrib.data import Iterator
from tensorflow.contrib.layers.python.layers import regularizers
from niftynet.layer.loss_segmentation import LossFunction
from util.data_loader import *
from util.train_test_func import *
from util.parse_config import parse_config
from util.MSNet import MSNet
from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins.hparams import api as hp


class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'MSNet':
            return MSNet
        # add your own networks here
        print('unsupported network:', name)
        exit()


def train(config_file, loss_type, lr, run_dir):
    # 1, load configuration parameters
    config = parse_config(config_file)
    config_data = config['data']
    config_net = config['network']
    config_train = config['training']

    random.seed(config_train.get('random_seed', 1))
    assert (config_data['with_ground_truth'])

    net_type = config_net['net_type']
    net_name = config_net['net_name']
    class_num = config_net['class_num']
    batch_size = config_data.get('batch_size', 5)

    # 2, construct graph
    full_data_shape = [batch_size] + config_data['data_shape']
    full_label_shape = [batch_size] + config_data['label_shape']
    x = tf.placeholder(tf.float32, shape=full_data_shape)
    w = tf.placeholder(tf.float32, shape=full_label_shape)
    y = tf.placeholder(tf.int64, shape=full_label_shape)

    w_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    b_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    net_class = NetFactory.create(net_type)
    net = net_class(num_classes=class_num,
                    w_regularizer=w_regularizer,
                    b_regularizer=b_regularizer,
                    name=net_name)
    net.set_params(config_net)
    predicty = net(x, is_training=True)
    proby = tf.nn.softmax(predicty)

    loss_func = LossFunction(n_class=class_num,loss_type=loss_type)
    loss = loss_func(predicty, y, weight_map=w)

    print('size of predicty:', predicty)

    # 3, initialize session and saver
    # lr = config_train.get('learning_rate', 1e-3)
    lr = lr

    opt_step = tf.train.AdamOptimizer(lr).minimize(loss)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    dataloader = DataLoader(config_data)
    dataloader.load_data()

    # 4, start to train
    print("Train started")
    summary_writer = tf.summary.FileWriter(run_dir)
    loss_file = config_train['model_save_prefix'] + "_loss.txt"
    start_it = config_train.get('start_iteration', 0)
    if start_it > 0:
        saver.restore(sess, config_train['model_pre_trained'])
    loss_list, temp_loss_list = [], []
    summary = tf.Summary()
    for n in range(start_it, 1000):
        train_pair = dataloader.get_subimage_batch()
        tempx = train_pair['images']
        tempw = train_pair['weights']
        tempy = train_pair['labels']
        opt_step.run(session=sess, feed_dict={x: tempx, w: tempw, y: tempy})

        if n % config_train['test_iteration'] == 0:
            batch_dice_list = []
            for step in range(config_train['test_step']):
                train_pair = dataloader.get_subimage_batch()
                tempx = train_pair['images']
                tempw = train_pair['weights']
                tempy = train_pair['labels']
                dice = loss.eval(feed_dict={x: tempx, w: tempw, y: tempy})
                batch_dice_list.append(dice)
            batch_dice = np.asarray(batch_dice_list, np.float32).mean()
            batch_dice = np.asarray([1,2,3], np.float32).mean()

            t = time.strftime('%X %x %Z')
            print(t, 'n', n, 'loss', batch_dice)
            loss_list.append(batch_dice)
            summary.value.add(tag='lr', simple_value=1)
            if loss_type == 'Dice':
                summary.value.add(tag='loss type', simple_value=0)
            if loss_type == 'CrossEntropy':
                summary.value.add(tag='loss type', simple_value=1)
            summary_writer.add_summary(summary, n)

            np.savetxt(loss_file, np.asarray(loss_list))

        if (n + 1) % config_train['snapshot_iteration'] == 0:
            saver.save(sess, config_train['model_save_prefix'] + "_{0:}.ckpt".format(n + 1))
    sess.close()


def tensor_board_log(config_file):
    # Tensorboard initialize
    logdir = "/home/hooshman/Documents/khanmhmdi/Brats_cascase_NN_source/brats17/log/"
    HP_LR = hp.HParam('learning_rate', hp.Discrete([0.001,0.005,0.01,0.1]))
    HP_LOSS = hp.HParam('loss_type', hp.Discrete(['Dice','CrossEntropy']))

    session_num = 0

    for num_units in HP_LR.domain.values:
        for loss in HP_LOSS.domain.values:
            hparams = {
                HP_LR: num_units,
                HP_LOSS : loss
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            train(config_file=config_file, loss_type=loss, lr=num_units,
                  run_dir='/home/hooshman/Documents/khanmhmdi'
                          '/Brats_cascase_NN_source/brats17/log/' + run_name)
            session_num += 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Number of arguments should be 2. e.g.')
        print('    python train.py config17/train_wt_sg.txt')
        exit()
    config_file = str(sys.argv[1])
    assert (os.path.isfile(config_file))
    tensor_board_log(config_file)