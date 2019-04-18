# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:37:20 2019

@author: Vo Thanh Phuong
"""

from gmm_ubm import gmmubm_model
import constants
import numpy as np
import pickle
import os

def split_array(arr, n_parts):
    result = []
    avg = len(arr) / float(n_parts)

    last = 0
    while last < len(arr):
        result.append(arr[int(last):int(last + avg)])
        last += avg

    return result

def ubm_subtask(path):
    path = constants.ROOT_FOLDER + path
    mfcc = get_mfcc_features(path, derivative=False)
    #mfcc = mfcc.transpose(0,2,1).reshape(mfcc.shape[0],39,)
    return mfcc

#from sklearn.mixture import GaussianMixture
#def train(num_epochs, batch_size):
#    data_path = np.loadtxt(constants.DEV_PATHS, dtype='str', delimiter=',')[:,1]
#    print(data_path.shape)
#  
#    data = []
#    init_ubm = True
#    model = gmmubm_model(n_components=512, max_iter=num_epochs, tol = 0.1)
#    #model = GaussianMixture(512)
#    
#    log_likelihood = [-np.Inf]
#    
#    for i in range(num_epochs):
#        np.random.shuffle(data_path)        
#        split_paths = split_array(data_path, 1000)
#        
#        n_iter = 0
#        n_data = 0
#        sum_likelihood = 0        
#        sum_paths = 0
#        
#        for j in range(len(split_paths)):
#                
#            print('(Extract) Epochs {}, n_paths = {}/{}, size = {}, start..'
#                      .format(i+1, j + 1, len(split_paths), len(split_paths[j])))
#            for k in range(len(split_paths[j])):       
#                mfcc = ubm_subtask(split_paths[j][k])               
#                if len(data) == 0:
#                    data = mfcc
#                else:
#                    data = np.concatenate((data, mfcc), axis = 0)    
#                
#                sum_paths += 1
#                if sum_paths % 100 == 0:
#                    print('==> Extract {}/{} ({}/{}) wav file, data length = {}'
#                          .format(sum_paths, len(data_path), k + 1, len(split_paths[j]), len(data)))
#            print('==> Extract {}/{} ({}/{}) wav file, data length = {}'
#                          .format(sum_paths, len(data_path), len(split_paths[j]), len(split_paths[j]), len(data)))
#            
#            # train mini batch
#            while len(data) > batch_size:
#                partial_data = data[0:batch_size]
#                data = data[batch_size:len(data)]                
#                likelihood = model.partial_fit_ubm(partial_data, init_ubm)
#                #model.fit(partial_data)
#                #likelihood = model.score(partial_data)
#                init_ubm = False
#    
#                sum_likelihood += likelihood
#                n_data += batch_size
#                n_iter += 1    
#                print('(Train model) Epochs {}, iter {}, size = {}, avg-likelihood = {}'
#                      .format(i, n_iter, batch_size, likelihood / batch_size))
#    
#    
#        if len(data) >= 5 * model.n_componens:
#            likelihood = model.partial_fit_ubm(data, init_ubm)
#            init_ubm = False
#            
#            n_data += len(data)
#            sum_likelihood += likelihood
#            n_iter += 1
#            
#            print('(Train model) Epochs {}, iter {}, size = {}, avg-likelihood = {}'
#                      .format(i + 1, n_iter, batch_size, likelihood))
#            data = []
#            
#        avg_likelihood = sum_likelihood / n_data 
#        print('GMM epochs = {}, avg-log_likehood = {}'.format(i + 1, avg_likelihood)) 
#        log_likelihood.append(avg_likelihood)           
#        if (np.absolute(avg_likelihood - log_likelihood[-1]) < model.tol):
#            break          
#        
#        filehandler = open('Epochs-' + str(i), 'wb')
#        pickle.dump(model, filehandler)
#        filehandler.close()
#        
#    print('GMM...Done')                        
                
if __name__ == "__main__":   
    train(50, 5000)