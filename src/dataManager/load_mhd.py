# -*- coding: utf-8 -*-
"""
Created on 11 January, 2018 @ 11:54 AM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: Insight_AI_BayLabs
License: 
"""

import SimpleITK as sitk


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

