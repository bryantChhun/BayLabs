# -*- coding: utf-8 -*-
"""
Created on 29 January, 2018 @ 11:31 AM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: Insight_AI_BayLabs
License: 
"""

def mcc():
    '''
    Calibration file that describes the np.pad values to align segmentation masks in 3d

    arrays for np.pad that adds zeroes in the following format:
    [(z-top, z-bottom), (y-top, y-bottom), (x-left, x-right)]
    :return:
    '''
    mcc = {}
    mcc['Patient1_ED'] = [(15,1),(1,1),(1,19)]
    mcc['Patient1_ES'] = [(15,1),(1,2),(1,20)]
    mcc['Patient2_ED'] = [(17,1),(1,4),(1,13)]
    mcc['Patient2_ES'] = [(16,1),(1,3),(1,16)]
    mcc['Patient3_ED'] = [(1,1),(8,1),(5,1)]
    mcc['Patient3_ES'] = [(1,5),(6,1),(5,1)]
    mcc['Patient4_ED'] = [(4,1),(1,7),(3,1)]
    mcc['Patient4_ES'] = [(3,1),(1,2),(5,1)]
    mcc['Patient5_ED'] = [(50,1),(1,4),(1,38)]
    mcc['Patient5_ES'] = [(40,1),(1,3),(1,38)]

    mcc['Patient6_ED'] = [(43,1),(1,7),(1,43)]
    mcc['Patient6_ES'] = [(36,1),(1,1),(1,33)]
    mcc['Patient7_ED'] = [(50,1),(1,1),(1,35)]
    mcc['Patient7_ES'] = [(45,1),(1,9),(1,38)]
    mcc['Patient8_ED'] = [(18,1),(1,6),(1,16)]
    mcc['Patient8_ES'] = [(17,1),(1,10),(1,20)]
    mcc['Patient9_ED'] = [(1,5),(1,1),(1,16)]
    mcc['Patient9_ES'] = [(1,5),(1,3),(1,18)]
    mcc['Patient10_ED'] = [(45,1),(1,5),(1,44)]
    mcc['Patient10_ES'] = [(47,1),(1,6),(1,44)]

    mcc['Patient11_ED'] = [(50,1),(1,5),(1,45)]
    mcc['Patient11_ES'] = [(48,1),(1,1),(1,45)]
    mcc['Patient12_ED'] = [(1,1),(1,1),(1,1)]
    mcc['Patient12_ES'] = [(1,1),(1,1),(1,1)]
    mcc['Patient13_ED'] = [(3,1),(1,1),(3,1)]
    mcc['Patient13_ES'] = [(3,1),(1,1),(4,1)]
    mcc['Patient14_ED'] = [(1,1),(1,1),(6,1)]
    mcc['Patient14_ES'] = [(1,1),(1,1),(6,1)]
    mcc['Patient15_ED'] = [(1,1),(1,1),(5,1)]
    mcc['Patient15_ES'] = [(1,1),(1,1),(3,1)]

    return mcc