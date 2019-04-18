# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:09:44 2019

@author: Vo Thanh Phuong
"""

import numpy as np
import constants
import h5py
from get_mfcc_features import get_mfcc_features

def ubm_subtask(path):
    path = constants.ROOT_FOLDER + path[1]
    mfcc = get_mfcc_features(path, derivative=False)
    #mfcc = mfcc.transpose(0,2,1).reshape(mfcc.shape[0],39,)
    return mfcc

def generate_features_for_gmm_ubm_model(name):
    # Using 39-dimensions features, 13 (MFCCs + delta + delta-delta)
    
    data_path = np.loadtxt(constants.DEV_PATHS, dtype='str', delimiter=',')
    print(data_path.shape)
    
    data = []      
    n_path = 100 #len(data_path)
    for i in range(n_path):
        if len(data) == 0:
            data = ubm_subtask(data_path[i])
        else:
            data = np.concatenate((data,ubm_subtask(data_path[i])))
            
        if (i + 1) % 100 == 0:
            print('Finished {} utterances. Collected {} MFCCs'.format(i + 1, len(data)))
            
        if (i + 1) % 10000 == 0:
            np.savez(constants.UBM_FEATURES + '{}.npz'.format(i + 1), data)
            data = []
    
    if (len(data) > 0):
        np.savez(constants.UBM_FEATURES + '{}.npz'.format(n_path), data)
    
    print('Finished')
    return

def generate_features_for_cnn_model():    
    return

if __name__ == "__main__":
    generate_features_for_gmm_ubm_model(constants.UBM_FEATURES)