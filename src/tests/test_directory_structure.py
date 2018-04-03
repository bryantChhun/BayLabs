# -*- coding: utf-8 -*-
"""
Created on 02 April, 2018 @ 5:21 PM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: BayLabs

License: 
"""

import os

import unittest

'''
Methods to check whether directory structure is acceptable (and data is placed in the right spots)
'''

class TestDirectoryStructure(unittest.TestCase):

    def test_data(self):
        self.assertIs(os.path.exists('./data'), True, msg='data folder does not exist or is not at highest level')
        patients = os.listdir('./data')
        for patient in patients:
            if os.path.isdir(patient):
                self.assertIs(os.path.exists('./data/' + str(patient) + '/'+str(patient)+'_ED_ES_time.txt'), True,
                              msg=str(patient)+'does not contain ED_ES_time.txt file')
                self.assertIs(os.path.exists('./data/' + str(patient) + '/'+str(patient)+'_ED_truth.vtk'), True,
                              msg=str(patient)+'does not contain ED ground truth .vtk file')
                self.assertIs(os.path.exists('./data/' + str(patient) + '/'+str(patient)+'_ES_truth.vtk'), True,
                              msg=str(patient)+'does not contain ES ground truth .vtk file')

    def test_images_npy(self):
        self.assertIs(os.path.exists('./images'), True, msg='images folder does not exist or is not at highest level')

    def test_masks_npy(self):
        self.assertIs(os.path.exists('./masks'), True, msg='masks folder does not exist or is not at highest level')

    def test_src(self):
        self.assertIs(os.path.exists('./src'), True, msg='src folder does not exist or is not at highest level')

def main():

    tester = TestDirectoryStructure()

    tester.test_data()
    tester.test_images_npy()
    tester.test_masks_npy()
    tester.test_src()

if __name__ == '__main__':
    main()