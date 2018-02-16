# -*- coding: utf-8 -*-
"""
Created on 05 February, 2018 @ 10:58 PM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: BayLabs
License: 
"""
import numpy as np
import re, os, glob
import src.dataManager.load_vtk as load_vtk
import src.dataManager.load_mhd as load_mhd
import src.preprocess.contour_preprocess as contour_preprocess
from src.preprocess import maskCentering


class preprocessData(object):

    def __init__(self, directory, frame):
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

        # get img frames for only ED and ES
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
            match = re.search("Patient(..)_frame", self.imgfile)
            self.patient_num = int(match.group(1))
        except AttributeError:
            try:
                match = re.search("Patient(.)_frame", self.imgfile)
                self.patient_num = int(match.group(1))
            except AttributeError:
                print("can't find patient num")

        self.prepare_images()
        self.prepare_masks()

    def prepare_images(self):
        self.image = load_mhd.mhd_to_array(self.imgfile)
        self.image_zdepth, self.image_height, self.image_width = self.image.shape
        self.image_spacing = load_mhd.mhd_to_spacing(self.imgfile)
        if not os.path.exists('./images'):
            os.makedirs('./images')
        np.save("./images/"+self.imgfile,self.image)
        return None

    def prepare_masks(self):
        '''
        method that loads vtk meshes, interpolates a binary mask, then writes to np array
        if patient is in "chooseindex", the mask is constructed with an extended volume
            to avoid clipping during interpolation
            - elements of "chooseindex" refer to patient #
            - REPROCESSED MASKS HAVE INCREASED X-Y DIMENSIONS
        :return:
        '''

        #check here for patient num
        chooseindex = [5, 6, 7, 11]
        if self.patient_num in chooseindex:
            extended_vol = True
        else:
            extended_vol = False

        # load vtk mesh and convert it to a 3d binary mask
        x, y, z = load_vtk.load_vtk(self.vtkfile).T
        space_z, space_y, space_x = self.image_spacing
        x, y, z = contour_preprocess.scale_contour(x, y, z, space_x, space_y, space_z)
        if self.patient_num in chooseindex:
            mask3d = contour_preprocess.contour_to_mask(x, y, z, self.image_width + 50, self.image_height + 50,
                                                        self.image_zdepth)
        else:
            mask3d = contour_preprocess.contour_to_mask(x, y, z, self.image_width, self.image_height, self.image_zdepth)

        # 3d alignment of masks based on calibration key
        calibrationKey = 'Patient'+str(self.patient_num)+'_'+self.masktype
        mask3d_centered = maskCentering.apply_recentering_mask3d(mask3d, calibrationKey=calibrationKey, extended_vol=extended_vol)

        if not os.path.exists('./masks'):
            os.makedirs('./masks')
        np.save('./masks'+self.vtkfile[:-4]+"_preprocess_centered.npy", mask3d_centered)
