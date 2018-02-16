#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 05 February, 2018 @ 11:16 PM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: BayLabs
License: 
"""


import glob
import src.preprocess.mask_preprocess as mp
import os

def imagePreprocess():
    '''
    method to generate images, binary-interpolated-aligned masks, and save them as npy arrays:

    1) load img.mhd, scale it, then save to directory './images'
    2) load mask.vtk
        a) preprocess with contour_preprocess
        b) align using apply_recentering_mask3d
        c) save to directory './masks'

    :return:
    '''

    data_dir = './data'

    glob_search = os.path.join(data_dir, "Patient*")
    patient_dirs = sorted(glob.glob(glob_search))
    if len(patient_dirs) == 0:
        raise Exception("No patient directories found in {}".format(data_dir))

    for patient_dir in patient_dirs:
        # each Patient folder contains approx 20 time frames of 3D images
        p = mp.preprocessData(patient_dir, frame="ED")
        p = mp.preprocessData(patient_dir, frame="ES")

if __name__ == '__main__':
    imagePreprocess()