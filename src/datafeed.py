# -*- coding: utf-8 -*-
"""
Created on 04 February, 2018 @ 11:32 AM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: BayLabs
License: 
"""

from __future__ import print_function

from skimage.transform import resize
import numpy as np
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
from random import randint
import os
import glob

from src import dataload


def create_generators(batch_size, train_num, test_num, shuffle=True, normalize_images=True,
                      augment_training=False,
                      validation_on=True,
                      augment_validation=False, augmentation_args={}):

    assert os.path.exists('./images'), 'images .npy folder does not exist!  must preprocess images first'
    assert os.path.exists('./masks'), 'images .npy folder does not exist!  must preprocess images first'

    #method to check num of patients in npy directory for images/masks
    imgs_search = os.path.join('./images', "P*_img.npy")
    imgs_files = glob.glob(imgs_search)
    if len(imgs_files) == 0:
        raise Exception("Couldn't find .npy preprocessed data in {}. "
                        "Wrong directory?".format('./images'))
    msks_search = os.path.join('./masks', "P*_msk.npy")
    msks_files = glob.glob(msks_search)
    if len(msks_files) == 0:
        raise Exception("Couldn't find .npy preprocessed data in {}. "
                        "Wrong directory?".format('./images'))

    assert (len(imgs_files) - train_num - test_num ) > 0, 'requested train/test split is too large for num of images available'
    assert (len(msks_files) - train_num - test_num) > 0, 'requested train/test split is too large for num of masks available'

    imgs_train, mask_train, imgs_test, mask_test = prepdata(train_size=train_num, test_size=test_num, tot_size=len(imgs_files), normalize=normalize_images)

    print('=' * 40)
    print('Creating data generator')
    print('=' * 40)

    if augment_training:
        idg = ImageDataGenerator(featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 **augmentation_args)

        #use idg.fit only if featurewise_center or featurewise_std_normalization are True
        #idg.fit(imgs_train)

        train_generator = idg.flow(imgs_train, mask_train,
                                   batch_size=batch_size, shuffle=shuffle)
        if validation_on and augment_validation:
            val_generator = idg.flow(imgs_test, mask_test,
                                     batch_size=batch_size, shuffle=shuffle)
        elif validation_on:
            idg = ImageDataGenerator()
            val_generator = idg.flow(imgs_test, mask_test,
                                     batch_size=batch_size, shuffle=shuffle)
        else:
            val_generator = None

        val_steps_per_epoch = ceil(len(imgs_test) / batch_size)
        train_steps_per_epoch = ceil(len(imgs_train) / batch_size)
    else:
        idg = ImageDataGenerator()
        train_generator = idg.flow(imgs_train, mask_train,
                                   batch_size=batch_size, shuffle=shuffle)
        if validation_on:
            val_generator = idg.flow(imgs_test, mask_test,
                                       batch_size=batch_size, shuffle=shuffle)
        else:
            val_generator = None

        val_steps_per_epoch = ceil(len(imgs_test) / batch_size)
        train_steps_per_epoch = ceil(len(imgs_train) / batch_size)

    return (train_generator, train_steps_per_epoch, val_generator, val_steps_per_epoch)


def prepdata(train_size, test_size, tot_size, flip_axes=[0,0], normalize=True):
    '''
    prepdata splits the training and testing size by patient.  This ensures that patient-specific data will NOT exist
       in both train and test sets. eg. Patient1 can exist in ONLY train OR test
    :param train_size: int for # of patients to include in train
    :param test_size: int for # of patients to include in train
    :param tot_size: int for total # of patients, used to determine holdout/eval set
    :param flip_axes: intended for ortho view.  can have the following values: [0,0] (unflipped), [0,1], [0,2]
    :param normalize: Boolean to determine whether to normalize image intensities.
    :return: imgs_train, mask_train, imgs_test, mask_test
    '''
    print('-' * 30)
    print('Loading train and validation data...')
    print('-' * 30)

    assert (train_size+test_size) <= tot_size, 'train + test sizes greater than num of patients!'
    eval_size = tot_size-(train_size+test_size)
    train_patients = []
    test_patients = []
    eval_patients = []

    while len(train_patients) < train_size:
        k = randint(1, 15)
        if k not in train_patients:
            train_patients.append(k)
    while len(test_patients) < test_size:
        k = randint(1, 15)
        if k not in train_patients and k not in test_patients:
            test_patients.append(k)
    while len(eval_patients) < eval_size:
        k = randint(1, 15)
        if k not in train_patients and k not in test_patients and k not in eval_patients:
            eval_patients.append(k)

    print('train patients = ' + str(train_patients))
    print('val patients = ' + str(test_patients))

    assert flip_axes==[0,0] or flip_axes==[0,1] or flip_axes == [0,2], 'not a valid flip_axes value.  must be 2-element list'

    #load all images into memory.
    imgs_train, mask_train = [], []
    imgs_test, mask_test = [], []
    for idx in train_patients:
        imgs_, msk_ = dataload.load_data(idx, 'ED', flip_axes=flip_axes)
        imgs_train += imgs_
        mask_train += msk_
        imgs_, msk_ = dataload.load_data(idx, 'ES', flip_axes=flip_axes)
        imgs_train += imgs_
        mask_train += msk_

    for idx in test_patients:
        imgs_, msk_ = dataload.load_data(idx, 'ED', flip_axes=flip_axes)
        imgs_test += imgs_
        mask_test += msk_
        imgs_, msk_ = dataload.load_data(idx, 'ES', flip_axes=flip_axes)
        imgs_test += imgs_
        mask_test += msk_

    # remove frames that contain no matching masks.
    imgs_train, mask_train = removeIndices(imgs_train, mask_train)
    imgs_test, mask_test = removeIndices(imgs_test, mask_test)

    # identify the max x-y dimensions
    imgs_train_shape = get_max_shape(imgs_train)
    mask_train_shape = get_max_shape(mask_train)
    imgs_test_shape = get_max_shape(imgs_test)
    mask_test_shape = get_max_shape(mask_test)

    max_y = np.max([imgs_train_shape[1], mask_train_shape[1], imgs_test_shape[1], mask_test_shape[1]])
    max_x = np.max([imgs_train_shape[2], mask_train_shape[2], imgs_test_shape[2], mask_test_shape[2]])

    # normalize shape of image list based on max x-y:
    imgs_train = reshape_list(imgs_train, (imgs_train_shape[0], max_y, max_x))
    mask_train = reshape_list(mask_train, (mask_train_shape[0], max_y, max_x))
    imgs_test = reshape_list(imgs_test, (imgs_test_shape[0], max_y, max_x))
    mask_test = reshape_list(mask_test, (mask_test_shape[0], max_y, max_x))

    # add channel dimension
    imgs_train = imgs_train[..., np.newaxis]
    imgs_test = imgs_test[..., np.newaxis]

    # one-hot encoding masks
    try:
        dims = mask_train.shape
        classes = len(set(mask_train.flatten()))  # get num classes from first image
        assert classes == 2, 'number mask train classes not equal 2'
        new_shape = dims + (classes,)
        mask_train = utils.to_categorical(mask_train).reshape(new_shape)
    except AssertionError:
        print('assertion error, num classes = ' + str(classes))

    try:
        dims = mask_test.shape
        classes = len(set(mask_test.flatten()))  # get num classes from first image
        assert classes == 2, 'number mask test classes not equal 2'
        new_shape = dims + (classes,)
        mask_test = utils.to_categorical(mask_test).reshape(new_shape)
    except AssertionError:
        print('assertion error, num classes = ' + str(classes))

    # image intensity normalization
    imgs_train = imgs_train.astype('float64')
    imgs_test = imgs_test.astype('float64')
    if normalize:
        #normalize train
        mean = np.mean(imgs_train)  # mean for data centering
        std = np.std(imgs_train)  # std for data normalization
        imgs_train -= mean
        imgs_train /= std
        #normalize test
        mean = np.mean(imgs_test)  # mean for data centering
        std = np.std(imgs_test)  # std for data normalization
        imgs_test -= mean
        imgs_test /= std

    mask_train = mask_train.astype('uint8')
    mask_test = mask_test.astype('uint8')

    return imgs_train, mask_train, imgs_test, mask_test


def removeIndices(imlist, msklist):
    '''
    this method removes images from both imlist and msklist that don't have corresponding masks.
        this can be as much as 50% of a z-stack
    :param imlist: list of 2d images(nparray)
    :param msklist: list of 2d masks(nparray)
    :return:imlist, msklist, truncated
    '''
    assert len(imlist) == len(msklist), 'imlist and msklist are not same length!'
    print("length of initial image list = "+str(len(imlist)))

    indexlist = []
    for idx, element in enumerate(msklist):
        # if np.max(element) == 0:
        #     indexlist += [idx]
        if np.count_nonzero(element) < 20:
            indexlist += [idx]

        else:
            pass

    print("length of new image list = "+str(len(imlist)-len(indexlist)))

    for i in sorted(indexlist, reverse=True):
        del imlist[i]
        del msklist[i]

    print("removed {} images".format(len(indexlist)))

    return imlist, msklist


def reshape_list(input_list, max_shape):
    '''
    Used to force all elements of input_list to have the same dimensionality
    :param input_list: list of nparrays.  arrays are each 2d images/masks without channel or one-hot encoding
    :param max_shape: shape of the maximum x-y dim determined by get_max_shape
    :return: new list with reshaped values.
    '''
    out_list = np.zeros(shape=max_shape, dtype=np.uint8)
    for i in range(len(input_list)):
        out_list[i] = resize(input_list[i], (max_shape[1], max_shape[2]), preserve_range=True, mode='constant')
    return out_list


def get_max_shape(input):
    '''
    :param input: list of arrays.  len(input) = number of arrays
    :return:tuple of (batch, max-y, max-x)
    '''
    max_x = 0
    max_y = 0
    for item in input:
        if item.shape[0] > max_y:
            max_y = item.shape[0]
        if item.shape[1] > max_x:
            max_x = item.shape[1]
    #forcing even for unet downsampling block
    if max_x%2 != 0:
        max_x +=1
    if max_y%2 != 0:
        max_y += 1
    print('shape 1 = {} , {}'.format(max_y, max_x))
    return (len(input), max_y, max_x)