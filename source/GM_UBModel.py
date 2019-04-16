# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:40:39 2019

@author: Vo Thanh Phuong
"""

from MFCCExtractor import mfcc_extractor
from GMModel import gm_model
import numpy as np

class gmmubm_model:
    def __init__(self, n_components=1, covariance_type='diag', tol = 1e-3, min_covar = 1e-3, max_iter=100, relevance_factor = 9):
        self.n_componens = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.min_covar = min_covar
        self.max_iter = max_iter
        self.gmm_ubm = gm_model(self.n_componens, self.covariance_type, self.tol,
                                self.min_covar,
                                self.max_iter)
        self.gmms = {}
        self.relavance_factor = relevance_factor
        
    def fit_ubm(self, data):
        self.gmm_ubm.fit(data)
        
    def enroll(self, id, data):
        self.gmms.update({id: self.gmm_ubm.clone()})
        model = self.gmms[id]
        
        log_likelihood = [-np.Inf]
        for step in range(self.max_iter):        
            
            detail = model.e_step(data)
            total = np.sum(detail, axis=0)
            
            for i in range(model.n_componens):
                cluster = model.clusters[i]
                alpha = total[i] / (total[i] + self.relavance_factor)
                mean = np.dot(detail[:,i].T, data) / total[i]
                
                cluster.mean = (1-alpha)*cluster.mean + alpha * mean
            
            likelihood = np.sum(np.log(model.pdf(data)))
            print('GMM it = {}, log_likehood = {}'.format(step + 1, likelihood))            
            if (np.absolute(likelihood - log_likelihood[-1]) < self.tol):
                break
            else:
                log_likelihood.append(likelihood)                
        print('GMM...Done')     
            
    def predict(self, data):
        score = -np.Inf
        result = None
        for i in self.gmms:
            gmm = self.gmms[i]
            if (gmm.score(data) > score):
                score = gmm.score(data)
                result = i
              
        return result