# Cardiac Ultrasound Segmentation

This repository contains code for segmenting the left ventricle of the heart from 4d ultrasound data


# Installation
Dependencies:

simpleitk = 1.0, scikit-image = 0.13, scipy = 0.19, vtk = 7.0, keras, tensorflow

Download the repo to your drive then run script commands from root directory.

# Data

As per agreement with the 2014 CETUS competition, data can't be shared on this repository.  Please go to the competition site, register, sign the data-NDA, then follow the instructions below.

https://www.creatis.insa-lyon.fr/Challenge/CETUS/

Place patient data folders in the /data directory, located at the top level of the repository (or create it). These folders should have a structure like:

    directory = "/data"
    		/data/Patient1
    		/data/Patient2
             	/data/Patient3
	     	...
             
      		/data/Patient1/Patient1_frame01.mhd
      		/data/Patient1/Patient1_frame02.mhd
      		...
      		/data/Patient1/Patient1_ED_ES_time.txt
      		/data/Patient1/Patient1_ED_truth.vtk
      		/data/Patient1/Patient1_ES_truth.vtk

Data must be preprocessed before running train.  There are several stages to the preprocessing:
1) Convert .vtk segmentation masks to binary masks (ground truth).
2) Parse timestamp to match .vtk masks and image data, use simple unsupervised methods to align the .vtk binary masks to data, then save as a calibration file.
3) Use calibration file to align masks/data, then save as .npy arrays for training.

From the root directory, run

	python image_preprocess.py

Any patient image files and .vtk masks in /data directory will be converted into .npy arrays.  Both will be saved to new directories:

	/images
	/masks

Running image preprocessing can take some time.

# Running models

To train a model, from the root directory run:

	python train.py -defaults.config
    
The defaults.config file contains flags for all training parameters.
