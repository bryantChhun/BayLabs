# -*- coding: utf-8 -*-
"""
Created on 02 April, 2018 @ 9:50 PM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: BayLabs
License: 
"""

import numpy as np

import unittest
import os
import glob

'''
Methods to check whether all IMAGE data in both raw and npy format are trainable
'''

class TestImageData(unittest.TestCase):

    def test_image_data(self):
        patients = os.listdir('./data')
        for patient in patients:
            #check whether any .mhd file exists in each patient folder
            if os.path.isdir(patient):
                self.assertIs(bool(glob.glob('./data/' + str(patient) + '/'+str(patient)+'*.mhd')), True,
                              msg=str(patient)+'data directory does not contain .mhd image file types')

        patients = os.listdir('./images')
        for patient in patients:
            #check whether any img.npy files exist in images folder
            self.assertIs(bool(glob.glob('./images/' + str(patient) + '*.npy')), True,
                          msg=str(patient) + 'image directory does not contain .npy image file types, try preprocessing data again')

    def test_image_format(self):
        patients = os.listdir('./images')
        for patient in patients:
            #check whether .npy files match correct image dimensionality
            pat = np.load('./images/'+str(patient))
            self.assertIs(bool(pat.shape[0] > 100), True, msg=str(patient)+'z-axis for patient is too small')
            self.assertIs(bool(pat.shape[1] > 100), True, msg=str(patient) + 'y-axis for patient is too small')
            self.assertIs(bool(pat.shape[2] > 100), True, msg=str(patient) + 'x-axis for patient is too small')


def main():
    tester = TestImageData()

    tester.test_image_data()
    tester.test_image_format()

if __name__ == '__main__':
    main()