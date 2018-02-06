# -*- coding: utf-8 -*-
"""
Created on 13 January, 2018 @ 11:08 AM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: Insight_AI_BayLabs
License: 
"""

import numpy as np
import vtk
from vtk import vtkPolyDataReader
from vtk.util import numpy_support as ns


def load_vtk(filename):
    # vtk files are polydata types.
    # we use primarily the "vtk_to_numpy" method in numpy_support
    reader = vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    nodes_vtk_array = ns.vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
    return nodes_vtk_array  #array is in format X, Y, Z

def write_numpy_to_vtk(input):
    # input is numpy data array
    # deep is like deepcopy
    VTK_data = ns.numpy_to_vtk(num_array=input.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    return VTK_data


# Convert numpy array to VTK array (vtkFloatArray)
def convert_to_vtk(input_array):
    vtk_data_array = ns.numpy_to_vtk(
    num_array=input_array.transpose(2, 1, 0).ravel(),  # ndarray contains the fitting result from the points. It is a 3D array
    deep=True,
    array_type=vtk.VTK_FLOAT)
    img_vtk = vtk.vtkPolyData()
    img_vtk.SetDimensions(input_array.shape)
    # img_vtk.SetSpacing(spacing[::-1])
    img_vtk.GetPointData().SetScalars(vtk_data_array)

# Convert the VTK array to vtkImageData
def convert_to_poly(vtk_data_array):
    #img_vtk = vtk.vtkImageData()
    img_vtk = vtk.vtkPolyData()
    img_vtk.SetDimensions(vtk_data_array.shape)
    #img_vtk.SetSpacing(spacing[::-1])
    img_vtk.GetPointData().SetScalars(vtk_data_array)




