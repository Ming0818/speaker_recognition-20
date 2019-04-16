# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:26:20 2019

@author: Vo Thanh Phuong
"""

import librosa
import constants
import speechpy
import numpy as np

def get_mfcc_features(wav_file):
    [signal,fs] = librosa.load(wav_file, constants.SAMPLE_RATE)
    signal *= 2**15

    mfcc = speechpy.feature.mfcc(signal, constants.SAMPLE_RATE,
                                 frame_length=0.025, frame_stride=0.01)
    mfcc = speechpy.feature.extract_derivative_feature(mfcc)
    
    return mfcc

def main():
    print('None')
    return

if __name__ == "__main__":
    main()