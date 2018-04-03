# -*- coding: utf-8 -*-
"""
Created on 02 April, 2018 @ 9:50 PM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: BayLabs
License: 
"""

import sys
import os
import glob
import numpy as np

import unittest

'''
Methods to check whether all MASK data in both raw and npy format are trainable
'''

class TestMaskData(unittest.TestCase):

    def test_mask_data(self):
        patients = os.listdir('./data')
        for patient in patients:
            #check whether any .vtk files exists in each patient folder
            if os.path.isdir(patient):
                self.assertIs(bool(glob.glob('./data/' + str(patient) + '/'+str(patient)+'*.vtk')), True,
                              msg=str(patient)+'data directory does not contain .vtk segmentation masks')

        patients = os.listdir('./masks')
        for patient in patients:
            #check whether any img.npy files exist in masks folder
            self.assertIs(bool(glob.glob('./masks/' + str(patient) + '*.npy')), True,
                          msg=str(patient) + 'masks directory does not contain .npy file types, try preprocessing data again')

    def test_mask_format(self):
        patients = os.listdir('./masks')
        for patient in patients:
            #check whether .npy files match correct image dimensionality
            pat = np.load('./masks/'+str(patient))
            self.assertIs(bool(pat.shape[0] > 100), True, msg=str(patient)+'z-axis for patient is too small')
            self.assertIs(bool(pat.shape[1] > 100), True, msg=str(patient) + 'y-axis for patient is too small')
            self.assertIs(bool(pat.shape[2] > 100), True, msg=str(patient) + 'x-axis for patient is too small')


def main():
    tester = TestMaskData()

    tester.test_mask_data()
    tester.test_mask_format()

if __name__ == '__main__':
    main()