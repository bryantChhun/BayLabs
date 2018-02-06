# -*- coding: utf-8 -*-
"""
Created on 30 January, 2018 @ 10:54 AM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: BayLabs
License: 
"""

from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread

import SimpleITK as sitk
import glob
import imageio
import re

'''
dataload.py contains methods for loading patient data into np arrays (.mhd and .vtk files)

it also contains methods for loading np arrays into lists, which are then fed into datafeed.py

preprocessing img/msk files into .npy arrays allows congruity of data types before training.
'''

def mhd_to_array(filename):
    # filename = 'file.mhd'
    img = sitk.ReadImage(filename)
    array = sitk.GetArrayFromImage(img)
    #img should be nparray of shape = (Z, Y, X)
    return array

def mhd_to_spacing(filename):
    img = sitk.ReadImage(filename)
    spacing = img.GetSpacing()
    #spacing should be same shape as images? = (Z, Y, X)
    return spacing


class PatientData(object):
    """
    Data directory structure (for 3d-US patient 01):
    directory = "./data/Patient1"
    directory/
      Patient1_frame01.mhd
      Patient1_frame02.mhd
      ...
      Patient1_ED_ES_time.txt
      Patient1_ED_truth.vtk
      Patient1_ES_truth.vtk

      Where ED_ and ES_truth are contour meshes of specific time points listed in the _time.txt file
    """
    def __init__(self, directory, frame='ED'):

        self.directory = os.path.normpath(directory)
        self.frame = frame

        # get ED ES time index from time file
        glob_search = os.path.join(directory, "P*time.txt")
        files = glob.glob(glob_search)
        if len(files) == 0:
            raise Exception("Couldn't find ED ES time file in {}. "
                            "Wrong directory?".format(directory))
        with open(files[0], 'r') as timefile:
            ED_line = timefile.readline().rstrip()
            ES_line = timefile.readline().rstrip()
        ED_tindex = int(ED_line.split('frame ')[-1])
        ES_tindex = int(ES_line.split('frame ')[-1])

        # get img, vtk(mask) frames for only ED and ES
        if frame == "ED":
            glob_search = os.path.join(directory, "P*_frame{:02d}.mhd".format(ED_tindex))
            self.imgfile = glob.glob(glob_search)[0]
            glob_search = os.path.join(directory, "P*_ED_truth.vtk")
            print("ED vtk file = "+str(glob_search))
            self.vtkfile = glob.glob(glob_search)[0]
        elif frame == "ES":
            glob_search = os.path.join(directory, "P*_frame{:02d}.mhd".format(ES_tindex))
            self.imgfile = glob.glob(glob_search)[0]
            glob_search = os.path.join(directory, "P*_ES_truth.vtk")
            print("ES vtk file = " + str(glob_search))
            self.vtkfile = glob.glob(glob_search)[0]
        else:
            raise Exception("time frame must be defined for ED or ES")

        try:
            match = re.search("Patient(..)", directory)
            self.patient_num = int(match.group(1))
        except AttributeError:
            try:
                match = re.search("Patient(.)", directory)
                self.patient_num = int(match.group(1))
            except AttributeError:
                print("can't find patient num")

    def load_images(self):
        self.image = mhd_to_array(self.imgfile)
        return None

    def load_masks(self):
        try:
            self.mask = np.load(self.vtkfile[:-4] + "_preprocess_centered.npy")
        except Exception as ex:
            print(ex)
            print('no preprocessed, centered mask exists')
            print('you must generate binary mask from vtk')
        return None


    def write_images_as_np(self, targetdir, frame):
        np.save(targetdir+'patient{}_{}_img'.format(self.patient_num, frame), self.image)

    def write_masks_as_np(self, targetdir, frame):
        np.save(targetdir + 'patient{}_{}_msk'.format(self.patient_num, frame), self.mask)

def write_patients_as_np(data_dir):
    '''
    Method that calls above PatientData class, loads patient data from directory
    and writes as numpy arrays for quick retrieval later.
    :param data_dir: location of folder containing "Patient1 ... Patient2 ... Patient3"
    :return: None
    '''

    glob_search = os.path.join(data_dir, "Patient*")
    patient_dirs = sorted(glob.glob(glob_search))
    if len(patient_dirs) == 0:
        raise Exception("No patient directories found in {}".format(data_dir))

    if not os.path.exists('./images'):
        os.makedirs('./images')
    if not os.path.exists('./masks'):
        os.makedirs('./masks')

    for patient_dir in patient_dirs:
        #each Patient folder contains approx 20 time frames of 3D images
        p = PatientData(patient_dir, frame="ED")
        p.load_images()
        p.load_masks()
        p.write_images_as_np('./images/', frame="ED")
        p.write_masks_as_np('./masks/', frame="ED")

        p = PatientData(patient_dir, frame="ES")
        p.load_images()
        p.load_masks()
        p.write_images_as_np('./images/', frame="ES")
        p.write_masks_as_np('./masks/', frame="ES")


def load_data(patient_index, frame, flip_axes):
    '''
    Method to load 3d patient data from .npy array.  Used for model training
    :param patient_index: patient number
    :param frame: ED or ES time points
    :param flip_axes: for training on orthogonal views: [0,1] flips Z and Y axis, [0,2] flips Z and X axis
    :return: list comprehension of imgs and masks
    '''
    imgs_train = np.load('./images/Patient{}_{}_img.npy'.format(patient_index, frame))
    mask_train = np.load('./masks/Patient{}_{}_msk.npy'.format(patient_index, frame))
    if flip_axes == [0,1]:
        imgs_train = np.swapaxes(imgs_train, flip_axes[0], flip_axes[1])
        mask_train = np.swapaxes(mask_train, flip_axes[0], flip_axes[1])
    elif flip_axes == [0,2]:
        imgs_train = np.rot90(np.swapaxes(imgs_train, flip_axes[0], flip_axes[1]), 3)
        mask_train = np.rot90(np.swapaxes(mask_train, flip_axes[0], flip_axes[1]), 3)
    else:
        print('no flip!')
    return [img for img in imgs_train], [msk for msk in mask_train]
