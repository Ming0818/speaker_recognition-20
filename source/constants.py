# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 00:38:39 2019

@author: Vo Thanh Phuong
"""

# File constants.py contains config of project

SAMPLE_RATE = 16000
WIN_LEN = 0.025
WIN_STEP = 0.01

ROOT_FOLDER = 'E:/Learning/HCMUS/2018-2019/Khoa Luan Tot Nghiep/Data/VoxCeleb1/wav/'
DEV_SPEAKERS = '../data/txt/dev_speakers.txt'
TEST_SPEAKERS = '../data/txt/test_speakers.txt'
DEV_PATHS = '../data/txt/dev_speakers_paths.txt'
TEST_PATHS = '../data/txt/test_speakers_paths.txt'
IDENTIFICATION_SPLIT = '../data/txt/iden_split.txt'
VERIFICATION_TEST = '../data/txt/veri_test.txt'

CNN_FEATURES = '../data/feats/cnn_features/'
UBM_FEATURES = '../data/feats/ubm_features/'