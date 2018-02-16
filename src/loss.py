#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 02 February, 2018 @ 5:19 PM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: BayLabs
License:
"""

from __future__ import division, print_function

from keras import backend as K

#======================================================
# methods similar to those in custom_metrics, but adjusted to be used as a loss

def precision_loss(y_true, y_pred, weights=[0.5,0.5]):
    batch_precision_coefs = _precision(y_true, y_pred, axis=[1, 2])
    precision_coefs = K.mean(batch_precision_coefs, axis=0)
    w = K.constant(weights) / sum(weights)
    return 1 - K.sum(w * precision_coefs)

def recall_loss(y_true, y_pred, weights=[0.5, 0.5]):
    batch_recall_coefs = _recall(y_true, y_pred, axis=[1, 2])
    recall_coefs = K.mean(batch_recall_coefs, axis=0)
    w = K.constant(weights) / sum(weights)
    return 1 - K.sum(w * recall_coefs)

def F_beta(y_true, y_pred, beta):
    batch_F_coefs = _F_beta(y_true, y_pred, axis=[1,2], beta=beta)
    F_coefs = K.mean(batch_F_coefs, axis=0)
    return 1-K.sum(F_coefs)

#======================================================

def _precision(y_true, y_pred, axis=None, smooth=1):
    true_positive = K.sum(y_true * y_pred, axis=axis)
    false_positive = K.sum(y_pred, axis=axis) - true_positive
    return (true_positive+smooth)/(true_positive+false_positive+smooth)

def _recall(y_true, y_pred, axis=None, smooth=1):
    true_positive = K.sum(y_true * y_pred, axis=axis)
    false_negative = K.sum(y_true, axis=axis) - true_positive
    return (true_positive+smooth)/(true_positive+false_negative+smooth)

def _F_beta(y_true, y_pred, axis=None, smooth=1, beta=1):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    true_positive = K.sum(y_true_int * y_pred_int, axis=axis)
    false_positive = K.sum(y_pred_int, axis=axis) - true_positive
    false_negative = K.sum(y_true_int, axis=axis) - true_positive
    prec = (true_positive+smooth)/(true_positive+false_positive+smooth)
    recall = (true_positive+smooth)/(true_positive+false_negative+smooth)
    return ( (1+beta**2) * prec * recall + smooth) / ( (beta**2)*prec + recall + smooth)


#======================================================

def soft_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=axis)
    area_true = K.sum(y_true, axis=axis)
    area_pred = K.sum(y_pred, axis=axis)
    return (2 * intersection + smooth) / (area_true + area_pred + smooth)
    
def hard_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    return soft_sorensen_dice(y_true_int, y_pred_int, axis, smooth)

sorensen_dice = hard_sorensen_dice

def sorensen_dice_loss(y_true, y_pred, weights=[0.5, 0.5]):
    # Input tensors have shape (batch_size, height, width, classes)
    # User must input list of weights with length equal to number of classes
    #
    # Ex: for simple binary classification, with the 0th mask
    # corresponding to the background and the 1st mask corresponding
    # to the object of interest, we set weights = [0, 1]
    # weights are usually useful for class imbalance.  Be mindful!
    batch_dice_coefs = soft_sorensen_dice(y_true, y_pred, axis=[1, 2])
    dice_coefs = K.mean(batch_dice_coefs, axis=0)
    w = K.constant(weights) / sum(weights)
    return 1 - K.sum(w * dice_coefs)

#======================================================

def soft_jaccard(y_true, y_pred, axis=None, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=axis)
    area_true = K.sum(y_true, axis=axis)
    area_pred = K.sum(y_pred, axis=axis)
    union = area_true + area_pred - intersection
    return (intersection + smooth) / (union + smooth)

def hard_jaccard(y_true, y_pred, axis=None, smooth=1):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    return soft_jaccard(y_true_int, y_pred_int, axis, smooth)

jaccard = hard_jaccard

def jaccard_loss(y_true, y_pred, weights):
    batch_jaccard_coefs = soft_jaccard(y_true, y_pred, axis=[1, 2])
    jaccard_coefs = K.mean(batch_jaccard_coefs, axis=0)
    w = K.constant(weights) / sum(weights)
    return 1 - K.sum(w * jaccard_coefs)

#======================================================

def weighted_categorical_crossentropy(y_true, y_pred, weights, epsilon=1e-8):
    ndim = K.ndim(y_pred)
    ncategory = K.int_shape(y_pred)[-1]
    # scale predictions so class probabilities of each pixel sum to 1
    y_pred /= K.sum(y_pred, axis=(ndim-1), keepdims=True)
    y_pred = K.clip(y_pred, epsilon, 1-epsilon)
    w = K.constant(weights) * (ncategory / sum(weights))
    # first, average over all axis except classes
    cross_entropies = -K.mean(y_true * K.log(y_pred), axis=tuple(range(ndim-1)))
    return K.sum(w * cross_entropies)
