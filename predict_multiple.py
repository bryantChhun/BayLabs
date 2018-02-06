# -*- coding: utf-8 -*-
"""
Created on 02 February, 2018 @ 1:40 PM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: BayLabs
License: 
"""

from src import predict
import os

def p_mult(target_directory, weights_dir, modeltype):
    '''
    take all .npy arrays from directory and run inference on each zplane
    :param directory:
    :return:
    '''

    end_patient = 5

    for patient_indx in range(1, end_patient+1):

        if not os.path.exists(target_directory+'/{}'.format(patient_indx)):
            os.makedirs(target_directory+'/{}'.format(patient_indx))
        predict.predict(patient_indx, img_directory='./',
                        target_directory = target_directory+'/{}'.format(patient_indx),
                        weights_directory=weights_dir, modeltype=modeltype)

        print("finished with patient {}".format(patient_indx))

    return None


if __name__ == '__main__':
    p_mult('./singleortho_dropout05_batchnorm_1_pred', './singleortho_dropout05_batchnorm_1/weights.120-0.4707-0.5296.hdf5', 'dilatedunet')