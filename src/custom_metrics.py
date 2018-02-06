# -*- coding: utf-8 -*-
"""
Created on 02 February, 2018 @ 5:19 PM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: BayLabs
License: 
"""

from keras import backend as K

#dice is same as F1 is same as Sorensen Dice Coefficient
def dice(y_true, y_pred):
    batch_dice_coefs = sorensen_dice_(y_true, y_pred, axis=[1, 2])
    dice_coefs = K.mean(batch_dice_coefs, axis=0)
    return dice_coefs[1]    # HACK for 2-class case

def jaccard(y_true, y_pred):
    batch_jaccard_coefs = jaccard_(y_true, y_pred, axis=[1, 2])
    jaccard_coefs = K.mean(batch_jaccard_coefs, axis=0)
    return jaccard_coefs[1] # HACK for 2-class case

def precision(y_true, y_pred, axis=[1,2], smooth=1):
    batch_precision_coefs = _precision(y_true, y_pred, axis=axis, smooth=smooth)
    precision_coefs = K.mean(batch_precision_coefs, axis=0)
    return precision_coefs[1]

def recall(y_true, y_pred, axis=[1,2], smooth=1):
    batch_recall_coefs = _recall(y_true, y_pred, axis=axis, smooth=smooth)
    recall_coefs = K.mean(batch_recall_coefs, axis=0)
    return recall_coefs[1]

def F1(y_true, y_pred, axis=[1,2], smooth=1):
    batch_recall_coefs = _F1(y_true, y_pred, axis=axis, smooth=smooth)
    recall_coefs = K.mean(batch_recall_coefs, axis=0)
    return recall_coefs[1]


# ======================================================

def _precision(y_true, y_pred, axis=None, smooth=1):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    #because y_true is binary, product is the same as intersection
    true_positive = K.sum(y_true_int * y_pred_int, axis=axis)
    false_positive = K.sum(y_pred_int, axis=axis) - true_positive
    return (true_positive+smooth)/(true_positive+false_positive+smooth)

def _recall(y_true, y_pred, axis=None, smooth=1):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    # because y_true is binary, product is the same as intersection
    true_positive = K.sum(y_true_int * y_pred_int, axis=axis)
    false_negative = K.sum(y_true_int, axis=axis) - true_positive
    return (true_positive+smooth)/(true_positive+false_negative+smooth)

def _F1(y_true, y_pred, axis=None, smooth=1):
    precision = _precision(y_true, y_pred, axis=None, smooth=1)
    recall = _recall(y_true, y_pred, axis=None, smooth=1)
    return ( 2*precision*recall ) / (precision + recall)

# ======================================================

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

jaccard_ = hard_jaccard

# ======================================================

def soft_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=axis)
    area_true = K.sum(y_true, axis=axis)
    area_pred = K.sum(y_pred, axis=axis)
    return (2 * intersection + smooth) / (area_true + area_pred + smooth)

def hard_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    return soft_sorensen_dice(y_true_int, y_pred_int, axis, smooth)

sorensen_dice_ = hard_sorensen_dice

# ======================================================

def weighted_categorical_crossentropy(y_true, y_pred, weights, epsilon=1e-8):
    ndim = K.ndim(y_pred)
    ncategory = K.int_shape(y_pred)[-1]
    # scale predictions so class probabilities of each pixel sum to 1
    y_pred /= K.sum(y_pred, axis=(ndim - 1), keepdims=True)
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
    w = K.constant(weights) * (ncategory / sum(weights))
    # first, average over all axis except classes
    cross_entropies = -K.mean(y_true * K.log(y_pred), axis=tuple(range(ndim - 1)))
    return K.sum(w * cross_entropies)
