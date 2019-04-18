# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:44:51 2019

@author: Vo Thanh Phuong
"""

from get_mfcc_features import get_mfcc_features
from gmm import gm_model
from gmm_ubm import gmmubm_model
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
import numpy as np
import pickle
import os

gm_models = []
gm_models_sklean = []
label = [ '1', '2', '3', '4', '5' ]

train_words = [ 'coffee', 'hello', 'laptop', 'mobile', 'music' ]
test_words =  [ 'speech' ]

ubm_data = []
all_data = {}

for i in label:    
    data = []
    for j in train_words:
        url = '../data/old_data/{}/{}.wav'.format(i,j)
        f = get_mfcc_features(url, derivative=False)
        
        if (len(data) == 0):
            data = f
        else:
            data = np.concatenate((data,f), axis = 0)
    
        if (len(ubm_data) == 0):
            ubm_data = f
        else:
            ubm_data = np.concatenate((ubm_data, f), axis = 0)

    all_data.update({i:data})

    # build background model
ubm_model_direct = '../data/old_data/ubmodel.mat'
if os.path.isfile(ubm_model_direct):
    filehandler = open(ubm_model_direct, 'rb')
    ub_gm_model = pickle.load(filehandler)
    filehandler.close()
else:
    ub_gm_model = gmmubm_model(n_components=6, max_iter=1000)
    ub_gm_model.fit_ubm(ubm_data)   
    filehandler = open(ubm_model_direct, 'wb')
    pickle.dump(ub_gm_model, filehandler)
    filehandler.close()

model = ub_gm_model.gmm_ubm
    # adapt MAP   
speakers_model = []
for i in all_data:
    speakers_model.append(ub_gm_model.enroll(i, all_data[i]))
    
num_test = len(label) * len(test_words)
num_true = 0

    # test
for i in label:    
    for j in test_words:
        url = '../data/old_data/{}/{}.wav'.format(i,j)
        f = get_mfcc_features(url, derivative=False)
        
        scores = [model.score(f) for model in speakers_model]
        if (str(np.argmax(scores) + 1) == i):
            print('{} {}'.format(np.max(scores), scores))
            num_true += 1
            
print('Accuracy = {}'.format(num_true/num_test))