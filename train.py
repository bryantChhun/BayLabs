#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 30 January, 2018 @ 10:41 AM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: Insight_AI_BayLabs
License: 
"""

from __future__ import print_function

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import optimizers
from keras import callbacks
import os
import src.cmdParser as cmd
import logging

from src import loss, datafeed
import src.models as models


class metricsHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('epoch'))
        self.losses.append(logs.get('loss'))
        self.losses.append(logs.get('dice'))
        self.losses.append(logs.get('val_loss'))
        self.losses.append(logs.get('val_dice'))

def train():

    logging.basicConfig(level=logging.INFO)

    args = cmd.opts.parse_arguments()

    logging.info("Loading dataset...")
    augmentation_args = {
        'rotation_range': args.rotation_range,
        'width_shift_range': args.width_shift_range,
        'height_shift_range': args.height_shift_range,
        'shear_range': args.shear_range,
        'zoom_range': args.zoom_range,
        'fill_mode': args.fill_mode,
    }
    train_generator, train_steps_per_epoch, val_generator, val_steps_per_epoch = datafeed.create_generators(
        args.batch_size, args.train_num, args.test_num,
        shuffle=args.shuffle,
        normalize_images=args.normalize,
        augment_training=args.augment_training,
        validation_on=args.validation_on,
        augment_validation=args.augment_validation,
        augmentation_args=augmentation_args)


    # ==========================================================
    # ======================Build model ========================


    print('=' * 40)
    print('Creating and compiling model...')
    print('=' * 40)
    imgs_train, mask_train = next(train_generator)
    _, height, width, channels = imgs_train.shape
    _, _, _, classes = mask_train.shape

    logging.info("Building model...")
    string_to_model = {
        "unet": models.unet,
        "dilated-unet": models.dilated_unet,
    }
    model = string_to_model[args.model]

    m = model(height=height, width=width, channels=channels, classes=classes,
              features=args.features, depth=args.depth, padding=args.padding,
              temperature=args.temperature, batchnorm=args.batchnorm,
              dropout=args.dropout)

    m.summary()

    #===============================================================
    #======================= Build metrics, lossfunc ===============

    if args.loss == 'pixel':
        def lossfunc(y_true, y_pred):
            return loss.weighted_categorical_crossentropy(y_true, y_pred, args.loss_weights)
    elif args.loss == 'dice':
        def lossfunc(y_true, y_pred):
            return loss.sorensen_dice_loss(y_true, y_pred, args.loss_weights)
    elif args.loss == 'jaccard':
        def lossfunc(y_true, y_pred):
            return loss.jaccard_loss(y_true, y_pred, args.loss_weights)
    else:
        raise Exception("Unknown loss ({})".format(args.loss))

    #===================== metrics =================================

    def dice(y_true, y_pred):
        batch_dice_coefs = loss.sorensen_dice(y_true, y_pred, axis=[1, 2])
        dice_coefs = K.mean(batch_dice_coefs, axis=0)
        return dice_coefs[1]

    def jaccard(y_true, y_pred):
        batch_jaccard_coefs = loss.jaccard(y_true, y_pred, axis=[1, 2])
        jaccard_coefs = K.mean(batch_jaccard_coefs, axis=0)
        return jaccard_coefs[1]

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)

    model.compile(loss=lossfunc, optimizer=sgd, metrics=['accuracy', dice, jaccard])

    # ========================================================
    # ==================  CallBacks     ======================

    print('=' * 40)
    print('Checkpoint config')
    print('=' * 40)

    checkpoint_folder = args.checkpoint_dir
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    if args.checkpoint:
        if args.loss == 'dice':
            filepath = os.path.join(
                args.outdir, "weights-{epoch:02d}-{dice:.4f}--{val_dice:.4f}.hdf5")
            monitor='val_dice'
            mode = 'max'
        elif args.loss == 'jaccard':
            filepath = os.path.join(
                args.outdir, "weights-{epoch:02d}-{jaccard:.4f}--{val_jaccard:.4f}.hdf5")
            monitor='val_jaccard'
            mode = 'max'
        checkpoint = ModelCheckpoint(
            filepath, monitor=monitor, verbose=1,
            save_best_only=True, mode=mode)
        callbacks = [checkpoint]
    else:
        callbacks = []

    history = metricsHistory()
    callbacks.append(history)
    # ========================================================
    # ========================================================

    print('=' * 40)
    print('Fitting model...')
    print('=' * 40)

    logging.info("Begin training.")
    model.fit_generator(train_generator, epochs=args.epochs,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_data = val_generator,
                        validation_steps=val_steps_per_epoch,
                        callbacks = callbacks)

    m.save(os.path.join(args.outdir, args.outfile))

    np.save(checkpoint_folder+"/losses", history.losses)



if __name__ == '__main__':
    train()
