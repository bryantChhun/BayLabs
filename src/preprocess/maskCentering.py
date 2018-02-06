# -*- coding: utf-8 -*-
"""
Created on 20 January, 2018 @ 11:28 AM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: Insight_AI_BayLabs
License: 
"""
'''
maskCentering.py contains methods for calculating the spatial shift between contour masks and its reference image.
All methods are based on smoothing using gaussian filters, plus cross-correlations between contour/image data

Should rename everything here to "contour" as this file deals only with contours, not masks
'''

import numpy as np
from scipy import signal
from scipy import ndimage
from src.preprocess import segment_skeleton as ss
import matplotlib.pyplot as plt
from src.dataManager.load_mhd import mhd_to_array
from src.preprocess.maskCenteringCalibration import mcc
import re, os, glob


def apply_recentering_mask3d(mskVolume, calibrationKey='Patient1_ED', extended_vol = False):
    '''

    :param mskVolume:
    :param calibrationKey:
    :param extended_vol:
    :return:
    '''
    mcc_dict = mcc()
    #out_vol = deepcopy(mskVolume)
    z_tup, y_tup, x_tup = mcc_dict[calibrationKey]
    if extended_vol == True:
        dyr = y_tup[0]+50
        dxr = x_tup[0]+50
    else:
        dyr = y_tup[0]
        dxr = x_tup[0]
    out_vol = np.pad(mskVolume,
                     mcc_dict[calibrationKey],
                     mode = 'constant',
                     constant_values=0)[z_tup[1]:-z_tup[0], y_tup[1]:-dyr, x_tup[1]:-dxr]
    return out_vol



def xcorr_2d(input1, input2):
    '''

    :param input1: target image (2d)
    :param input2: gaussian filtered contour (2d)
    :return: cross correlation deltas between target and filtered contour
    '''

    corr = signal.correlate(input1, input2, mode='same')

    #x-y coordinates of correlation vs center of image
    new_center_y, new_center_x = np.unravel_index(corr.argmax(), corr.shape)
    height, width = input1.shape
    img_center_y = height/2
    img_center_x = width/2
    #delta is (X, Y)
    delta = (img_center_x - new_center_x, img_center_y - new_center_y)
    return delta

def recenter_mask2d(msk, dx, dy):
    ''' This method takes scaled 2d contour

    :param msk: scaled 2d mask, but not recentered
    :param dx: output of 'getMaskCentering'
    :param dy: output of 'getMaskCentering'
    :return: mask that is padded with zeros then shifted
    '''
    if dx<=0 and dy<=0:
        shift_msk = np.pad(msk, [(0,abs(dx)), (0,abs(dy))], mode='constant')[abs(dx):, abs(dy):]
    elif dx>=0 and dy<=0:
        shift_msk = np.pad(msk, [(abs(dx),0), (0,abs(dy))], mode='constant')[:-abs(dx), abs(dy):]
    elif dx<=0 and dy>=0:
        shift_msk = np.pad(msk, [(0,abs(dx)), (abs(dy),0)], mode='constant')[abs(dx):, :-abs(dy)]
    elif dx>=0 and dy>=0:
        shift_msk = np.pad(msk, [(abs(dx),0), (abs(dy),0)], mode='constant')[:-abs(dx), :-abs(dy)]
    assert msk.shape == shift_msk.shape, 'padded/shifted mask is not the same dimension as input mask!'

    return shift_msk

def recenter_mask_3d(msk, dx, dy, dz):
    ''' This method takes scaled 2d contour

    :param msk: scaled 3d mask.  shape = (z, y, x)
    :param dx: output of 'getMaskCentering'
    :param dy: output of 'getMaskCentering'
    :return: mask that is padded with zeros then shifted
    '''
    if dx<=0 and dy<=0:
        shift_msk = np.pad(msk, [(0,0), (0,abs(dy)), (0,abs(dx))], mode='constant', constant_values=0)[:, abs(dy):, abs(dx):]
    elif dx>=0 and dy<=0:
        shift_msk = np.pad(msk, [(0,0), (0,abs(dy)), (abs(dx),0)], mode='constant', constant_values=0)[:, abs(dy):, :-abs(dx)]
    elif dx<=0 and dy>=0:
        shift_msk = np.pad(msk, [(0,0), (abs(dy),0), (0,abs(dx))], mode='constant', constant_values=0)[:, :-abs(dy), abs(dx):]
    elif dx>=0 and dy>=0:
        shift_msk = np.pad(msk, [(0,0), (abs(dy),0), (abs(dx),0)], mode='constant', constant_values=0)[:, :-abs(dy), :-abs(dx)]

    if dz<=0:
        shift_msk = np.pad(shift_msk, [(0,abs(dz)), (0,0), (0,0)], mode='constant', constant_values=0)[abs(dz):, :, :]
    elif dz>0:
        shift_msk = np.pad(shift_msk, [(abs(dz),0), (0,0), (0,0)], mode='constant', constant_values=0)[:-abs(dz), :, :]

    assert msk.shape == shift_msk.shape, 'padded/shifted mask is not the same dimension as input mask!'

    return shift_msk

def center_all_masks(image_directory):
    '''
    Method to convert all 3d volume masks into binary masks and center them based on calibration file
    chooseindex specifies masks whose x-y dimensions were extended by 50 pixels to prevent mask clipping
        these must be shifted by 50 more pixels than other masks.
        this condition is flagged with keyword=extended_vol
    :param image_directory:
    :return:
    '''

    chooseindex = [5, 6, 7, 11]

    glob_search = os.path.join(image_directory, "P*")
    files = glob.glob(glob_search)
    for patient_folder in files:
        try:
            match = re.search("Patient(..)", patient_folder)
            patient_num = int(match.group(1))
        except AttributeError:
            try:
                match = re.search("Patient(.)", patient_folder)
                patient_num = int(match.group(1))
            except AttributeError:
                print("can't find patient num")
        print("patient num = "+str(patient_num))

        if patient_num in chooseindex:
            extended_vol = True
        else:
            extended_vol = False

        calibrationKey_pat = 'Patient{}_'.format(patient_num)

        vol_search = os.path.join(patient_folder, "P*_truth_preprocess.npy")
        vol_files = glob.glob(vol_search)
        for numpy_mask in vol_files:
            try:
                match = re.search("_(..)_truth_preprocess.npy", numpy_mask)
                ED_ES = match.group(1)
            except AttributeError:
                print("can't find ED/ES value")
            calibrationKey = calibrationKey_pat + ED_ES
            mask3d = np.load(numpy_mask)

            #check if binary
            if len(set(mask3d.flatten())) != 2:
                print('mask volume is not binary! converting now')
                for idx, zplane in enumerate(mask3d):
                    # force masks to be binary, or one-hot will fail.
                    zplane = (zplane > 0.0).astype(int)
                    mask3d[idx] = zplane

            #recenter and reshape
            mask3d = apply_recentering_mask3d(mask3d, calibrationKey=calibrationKey, extended_vol=extended_vol)
            np.save(patient_folder+"/"+calibrationKey+"_truth_preprocess_centered.npy", mask3d)
        print("new patient")

    return None


################
# Methods below search the space for optimal x-y-z shift
# not implemented for production
################

def getMaskCentering(img, msk, msk_sigma=2, skeleton_threshold = 70, skeleton_sigma = 2):
    '''
    returns x-y shift needed to center the mask on the object in the image.
    img and msk must be 2d because skeleton works only on 2d
    Method:
        1) gaussian filter the 2d mask.  Should be only the boundary
        2) generate skeleton of image based on threshold, then gaussian filter
        3) normalize both gaussian filtered mask and skeleton to 255
        4) cross correlate to find shift

    :param img: 2d plane of image volume
    :param msk: 2d plane of contour mask scaled as per metadata.
    :param img_sigma: sigma of gaussian to filter the image skeleton
    :param skeleton_threshold: intensity threshold to calculate skeleton
    :param skeleton_sigma: sigma of gaussian to filter the skeleton
    :return: delta x, delta y of the shift
    '''
    assert img.shape == msk.shape, 'image and mask are not same dimensions'
    g_msk = ndimage.filters.gaussian_filter(msk, sigma=msk_sigma)
    norm_g_msk = 255 * (g_msk - np.min(g_msk)) / max(1, (np.max(g_msk) - np.min(g_msk)))
    skele = ss.to_skeleton(ss.to_binary(img, threshold=skeleton_threshold))
    g_skele = ndimage.filters.gaussian_filter(skele, sigma=skeleton_sigma)
    norm_g_skele = 255 * (g_skele - np.min(g_skele)) / max(1, (np.max(g_skele) - np.min(g_skele)))

    # x-correlation of both planes
    dx, dy = xcorr_2d(norm_g_skele, norm_g_msk)
    return int(dx), int(dy)

def scan_corr(input1, input2, sigma=2, skeleton=False, skeleton_threshold = 70, skeleton_sigma = 2):
    '''
    scans through all zplanes and correlates input1 with input2's z plane
    use this to plot dx and dy as a function of zplane for a better sense of z_max.
    :param input1: 3d image Volume
    :param input2: 3d mask Volume, without gaussian filtering
    :param sigma:       gaussian sigma
    :param skeleton:    whether input1 should be converted to skeleton
    :return: two lists: dx = deltax, dy= deltay
    '''
    xcorr_list = []
    for idx, zplane in enumerate(input1):
        #mask is calculated for every zplane
        g_input2 = ndimage.filters.gaussian_filter(input2[idx], sigma=sigma)
        norm_g_input2 = 255 * (g_input2 - np.min(g_input2)) / max(1, (np.max(g_input2) - np.min(g_input2)))
        #image is calculated for every zplane
        if skeleton == True:
            scaledSkele = ss.to_skeleton(ss.to_binary(zplane, threshold=skeleton_threshold))
            zplane = ndimage.filters.gaussian_filter(scaledSkele, sigma=skeleton_sigma)
        norm_zplane = 255 * (zplane - np.min(zplane)) / max(1, (np.max(zplane) - np.min(zplane)))
        #x-correlation of both planes
        xcorr_list.append(xcorr_2d(norm_zplane, norm_g_input2))

    dx, dy = np.array(xcorr_list)[:,0], np.array(xcorr_list)[:,1]
    return dx, dy

def scan_corr_findz(input1, input2, input2z, sigma=2, skeleton=False, skeleton_sigma = 2):
    '''
    scans through input1's volume, but compares to only one z plane in input2
    this method should output an optimal z shift
    for thorough testing, try iterating through several input2z planes
    :param input1:
    :param input2:
    :param sigma:
    :return:
    '''
    xcorr_list = []
    #mask is calculated for only one zplane specified by index=input2z
    g_input2 = ndimage.filters.gaussian_filter(input2[input2z], sigma=sigma)
    norm_g_input2 = 255 * (g_input2 - np.min(g_input2)) / max(1, (np.max(g_input2) - np.min(g_input2)))
    for idx, zplane in enumerate(input1):
        #image is calculated for every zplane
        if skeleton == True:
            skele = ss.to_skeleton(ss.to_binary(zplane, threshold=skeleton_threshold))
            zplane = ndimage.filters.gaussian_filter(skele, sigma=skeleton_sigma)
        norm_zplane = 255 * (zplane - np.min(zplane)) / max(1, (np.max(zplane) - np.min(zplane)))
        #x-correlation of both planes
        xcorr_list.append(xcorr_2d(norm_zplane, norm_g_input2))

    dx, dy = np.array(xcorr_list)[:, 0], np.array(xcorr_list)[:, 1]
    return dx, dy