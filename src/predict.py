# -*- coding: utf-8 -*-
"""
Created on 30 January, 2018 @ 12:19 PM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: testing
License: 
"""
from __future__ import print_function

import os
#from skimage.transform import resize
import numpy as np

from src import dataload
from src.models import dilatedunet, convunet

import imageio

smooth = 1.

def predict(patient_num = 1, img_directory='./', target_directory = './', weights_directory='./', modeltype='convunet'):

    patient_num = patient_num
    frame_ = ['ED','ES']
    flip_axes = [0,1]

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)

    imgs_pred, msks_pred = dataload.load_data(patient_index=patient_num, frame=frame_[0], flip_axes=flip_axes)

    imgs_pred = np.array(imgs_pred).astype('float64')
    mean = np.mean(imgs_pred)  # mean for data centering
    std = np.std(imgs_pred)  # std for data normalization
    imgs_pred -= mean
    imgs_pred /= std

    imgs_pred = np.array(imgs_pred)[:, :, :, None]

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    _, height, width, channels = imgs_pred.shape
    #_, _, _, classes = np.array(msks_pred).shape
    classes = 2

    if modeltype == 'convunet':
        model = convunet.unet(height=height, width=width, channels=channels, classes=classes,
                              features=32, depth=3, padding='same',
                              temperature=1, batchnorm=False,
                              dropout=0.5)

    elif modeltype == 'dilatedunet':
        model = dilatedunet.dilated_unet(height=height, width=width, channels=channels, classes=classes,
                                         features=32, depth=3, padding='same',
                                         temperature=1, batchnorm=False,
                                         dropout=0.5)
    else:
        print("no model type exists!")

    # ========================================================
    # ========================================================

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)


    model.load_weights(weights_directory)


    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)


    pred_dir = target_directory

    imgs_mask_pred = model.predict(imgs_pred, verbose=1)
    np.save(pred_dir+'/imgs_test.npy', imgs_mask_pred)


    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)


    for idx, image in enumerate(imgs_mask_pred):
        max = np.max(image[:,:,0])
        min = np.min(image[:,:,0])
        #image = (255*(image[:, :, 0]-min)/(max-min)).astype(np.uint8)
        imageio.imsave(os.path.join(pred_dir, 'prediction_0_' + str(idx) + '_pred.jpg'), image[:,:,0])

    for idx, image in enumerate(imgs_mask_pred):
        max = np.max(image[:,:,1])
        min = np.min(image[:,:,1])
        #image = (255*(image[:, :, 1]-min)/(max-min)).astype(np.uint8)
        imageio.imsave(os.path.join(pred_dir, 'prediction_1_' + str(idx) + '_pred.jpg'), image[:,:,0])


if __name__ == '__main__':
    predict()
