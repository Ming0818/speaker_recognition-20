# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:26:20 2019

@author: Vo Thanh Phuong
"""

import librosa
import constants
import speechpy
import numpy as np

def get_mfcc_features(wav_file, num_cepstral = 13, derivative = True, cmvn_size = 0):    
    # Read signal wav from file
    [signal,fs] = librosa.load(wav_file, constants.SAMPLE_RATE, mono=True)
    signal *= 2**15

    # Extract MFCCs features
    mfcc = speechpy.feature.mfcc(signal, constants.SAMPLE_RATE,
                                 frame_length=0.025, frame_stride=0.01,
                                 num_cepstral=num_cepstral)        
    
    # Cepstral Mean Variance Normalization
    if cmvn_size == 0:
        mfcc = speechpy.processing.cmvn(mfcc, True)
    elif cmvn_size > 0:
        mfcc = speechpy.processing.cmvnw(mfcc, cmvn_size, True)
    
    # Add delta and delta-delta features
    if derivative == True:
        mfcc = speechpy.feature.extract_derivative_feature(mfcc)
    
    return mfcc

def main():
    mfcc = get_mfcc_features('../data/wav/test.wav')
    print(mfcc[0])
    return

if __name__ == "__main__":
    main()