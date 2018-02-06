# -*- coding: utf-8 -*-
"""
Created on 05 February, 2018 @ 10:42 PM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: BayLabs
License: 
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator as plinear


def normalize_contour(x, y, z, space_x, space_y, space_z):
    '''
    The spacing values must be parsed out from the .mhd image files, NOT the .vtk meshes
    :param x:
    :param y:
    :param z:
    :return:
    '''
    return x/space_x,  y/space_y, z/space_z


def downsample_contour(x, y, z):
    '''
    round coordinate value to the nearest integer
    necessary for assignment to 3d array
    :param x:
    :param y:
    :param z:
    :return:
    '''
    x = np.asarray(list(map(round, x)), dtype=np.int)
    y = np.asarray(list(map(round, y)), dtype=np.int)
    z = np.asarray(list(map(round, z)), dtype=np.int)
    return x, y, z

def contour_to_mask(x, y, z, width, height, zdepth):
    '''
    method to be applied before downsample contour.
    Using scipy's LinearNDInterpolator, determine a convex hull that describes surface,
        LinearInterpolator returns zero (user defined), if input coords are outside hull
    Loop pixel-wise to assign values (more clever way probably exists!)
    About 20-30 mins per 200x200x200 array

    :param x: array of vtk x coords
    :param y: array of vtk y coords
    :param z: array of vtk z coords
    :param width: target image width
    :param height: target image height
    :param zdepth: target image zdepth
    :return: binary mask np volume
    '''
    coords = np.array(list(zip(z, y, x)))
    values = np.ones(shape=(coords.shape[0]))
    vtk_lp = plinear(coords, values, 0)
    coord_array = np.zeros(shape=(zdepth, height, width))

    for idx1, plane in enumerate(coord_array):
        for idx2, row in enumerate(plane):
            for idx3, column in enumerate(row):
                coord_array[idx1][idx2][idx3] = vtk_lp(idx1,idx2,idx3)
    print('new mask')
    return coord_array