# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 00:41:57 2019

@author: Vo Thanh Phuong
"""

from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvnpdf
import numpy as np
import copy

#Based on https://github.com/stober/gmm

class gm_cluster:
    def __init__(self, prior, data, min_covar):
        self.prior = prior
        self.mean = np.mean(data, axis=0)
        self.min_covar = min_covar
        self.dim = len(self.mean)
        self.update_covariance(np.cov(data, rowvar=0))

    def update_covariance(self, covariance):
        self.covariance = covariance
        self.covariance += self.min_covar * np.eye(self.dim)
        
        det = np.fabs(np.linalg.det(self.covariance))
        self.inv_covariance = np.linalg.inv(self.covariance)
        self.factor = (2.0 * np.pi)**(self.dim / 2.0) * (det)**(0.5)

    def pdf(self, x):
        tmp = x - self.mean        
        return np.exp(-0.5 * np.dot(np.dot(tmp,self.inv_covariance), tmp)) / self.factor

class gm_model:
    def __init__(self, n_components=1, covariance_type='full', tol = 1e-3, min_covar = 1e-3, max_iter=100):
        self.n_componens = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.min_covar = min_covar
        self.max_iter = max_iter
   
    def gmm_init_kmeans(self, data):
        label = (KMeans(self.n_componens, random_state=999)).fit(data).labels_
        clusters = []
        for i in range(self.n_componens):
            sub_data = data[label == i] 
            prior = len(sub_data) / len(label)               
            if len(sub_data) == 1:
                x = data[np.random.choice(range(len(data)), 5)]
                sub_data = np.concatenate((sub_data, x), axis = 0)
            clusters.append(gm_cluster(prior, sub_data, 
                                       self.min_covar))
        return clusters
    
    def clone(self):
        return copy.deepcopy(self)
    
    def e_step(self, data):             
        N = len(data)
        detail = np.zeros((N, self.n_componens))
        for i in range(N):
            total = 0
            for j in range(self.n_componens):
                comp = self.clusters[j]
                detail[i,j] = comp.prior * comp.pdf(data[i,:])
                total += detail[i,j]
            #detail[i,:] = detail[i,:] / np.sum(detail[i, :], axis = 0)
            detail[i,:] /= total
        return detail
    
    def m_step(self, data, detail):
        N = len(data)
        total = np.sum(detail, axis=0)
        for i in range(self.n_componens):
            comp = self.clusters[i]
            comp.mean = np.dot(detail[:,i].T, data) / total[i]
            covariance = np.zeros((comp.dim, comp.dim))
            for j in range(N):
                dx = data[j,:] - comp.mean
                covariance += detail[j,i] * np.outer(dx,dx)
            covariance /= total[i]            
            comp.update_covariance(covariance)
            #if self.covariance_type == 'diag':
            #    comp.covariance = np.diag(comp.covariance)
            comp.prior = total[i] / N

        return
    
    def partial_fit(self, data, init = False):
        if init == True:
            self.clusters = self.gmm_init_kmeans(data)
            
        detail = self.e_step(data)                
        self.m_step(data, detail)
            
        likelihood = np.sum(np.log(self.pdf(data)))
        return likelihood
    
    def fit(self, data):
        self.clusters = self.gmm_init_kmeans(data)
        log_likelihood = [-np.Inf]
        for step in range(self.max_iter):
            detail = self.e_step(data)                
            self.m_step(data, detail)
            
            likelihood = np.sum(np.log(self.pdf(data)))
            
            print('GMM it = {}, log_likehood = {}'.format(step + 1, likelihood))            
            if (np.absolute(likelihood - log_likelihood[-1]) < self.tol):
                break
            else:
                log_likelihood.append(likelihood)                
        print('GMM...Done')                        
        return
    
    def pdf(self, data):
        result = np.zeros((len(data), 1))
        for i in range(len(data)):
            for j in range(self.n_componens):            
                comp = self.clusters[j]
                result[i] += comp.prior * comp.pdf(data[i, :])
        return result
    
    def score(self, data):
        return np.sum(np.log(self.pdf(data))) / len(data)
            
    def mean(self):
        mean = np.zeros((1,self.dim))
        for i in range(self.n_componens):
            cluster = self.clusters[i]
            mean = mean + cluster.prior * cluster.mean
        return mean
    
    def covariance(self):
        m = self.mean()
        s = -np.outer(m,m)

        for i in range(self.n_componens):
            cluster = self.clusters[i]
            cm = cluster.mean
            cvar = cluster.covariance
            s += cluster.prior * (np.outer(cm,cm) + cvar)

        return s
        
#model = gm_model(n_components=2, min_covar=1e-4, max_iter=100)
#data = np.array([[1,2,3],[1,2,3], [1.1, 2.1, 3.1], [3,4,5], [3,4,5]])
#model.fit(data)
    
#print(model.score(data))