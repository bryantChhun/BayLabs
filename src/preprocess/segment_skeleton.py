# -*- coding: utf-8 -*-
"""
Created on 21 January, 2018 @ 5:37 PM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: Insight_AI_BayLabs
License: 
"""

from skimage.morphology import skeletonize
from copy import deepcopy


def to_binary(image, threshold):
    binim = deepcopy(image)
    binim[binim < threshold] = 0
    binim[binim >= threshold] = 1
    return binim

def to_skeleton(binim, scale = 255):
    skeleton = skeletonize(binim)
    return scale*skeleton
