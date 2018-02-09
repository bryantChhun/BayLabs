# Cardiac Ultrasound Segmentation

This repository contains code for segmenting the left ventricle of the heart from 4d ultrasound data


# Installation
Dependencies:

simpleitk, scikit-image, scipy, vtk, keras, tensorflow

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

Data must be preprocessed before running train.  From the root directory, run

	python image_preprocess.py

Any image files in data/Patient# directory will be convetred into .npy arrays.  Any .vtk masks in data/Patient# will be converted into binary masks, then into .npy arrays.  Both will be saved to new directories:

		/images
		/masks

Running image preprocessing can take as long as 30 minutes PER MASK.

# Running models

To train a model, from the root directory run:

	python train.py -defaults.config
    
The defaults.config file contains flags for all training parameters.
