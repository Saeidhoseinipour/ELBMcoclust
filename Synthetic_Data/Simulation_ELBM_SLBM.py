#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""SimulationELBM"""   # Simulate data from Exponential Latent Block Model based on X / beta_hat = R A_alpha C^T + E

# Author: Hoseinipour Saeid <saeidhoseinipour@aut.ac.ir> 


import numpy as np
from scipy.stats import multinomial
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import beta
from scipy.stats import bernoulli
from scipy.stats import poisson
from itertools import zip_longest

class rvs:
   

    def __init__(self, m = 100, n = 100, g = 2, s = 2, pi = None, rho = None):

        self.m = m
        self.n = n
        self.g = g
        self.s = s
        self.pi = pi
        self.rho = rho
        self.R = None 
        self.C = None
        self.A_alpha = None
        self.X_datamatrix = None
        self.rowcluster_matrix = None
        self.columncluster_matrix = None
        self.true_row_labels_X = None
        self.true_column_labels_X = None



    def GenerateELBM(self, alpha, model):                
        
        R = np.zeros((self.m, self.g),dtype=int)
        C = np.zeros((self.n, self.s),dtype=int)
        E_mn = np.ones((self.m, self.n))
 
        for i,j in zip_longest(range(self.m), range(self.n)):
            if i is not None:
                R[i,:] = multinomial.rvs(1, self.pi, random_state=None)
            if j is not None:
                C[j,:] = multinomial.rvs(1, self.rho, random_state=None)
        R_sort = np.zeros_like(R)
        R_sort[np.arange(len(R)),np.sort(np.argmax(R,axis=1))] = 1
        C_sort = np.zeros_like(C)
        C_sort[np.arange(len(C)),np.sort(np.argmax(C,axis=1))] = 1
        X_datamatrix = np.zeros((self.m,self.n))
        E_noise = np.zeros((self.m,self.n))
        A_alpha = self.A_alpha
        if (model == "Poisson"):
            A_alpha = np.log(alpha)
            X_datamatrix = R@A_alpha@C.T
            beta_hat =  X_datamatrix@E_mn.T@X_datamatrix
        elif (model == "Normal"):
            sigma = 10^2        
            A_alpha = alpha/sigma             # mu/sigma^2
            X_datamatrix = R@A_alpha@C.T
            beta_hat =  E_mn
        elif (model == "Bernoulli"):  
            A_alpha = np.log(alpha/(1-alpha))
            X_datamatrix = R@A_alpha@C.T   
            beta_hat =  E_mn                     
        elif (model == "Beta"): 
            A_alpha = alpha
            X_datamatrix = R@A_alpha@C.T
            beta_hat =  E_mn
        else:
            print("Model name not found")

        self.X_datamatrix = X_datamatrix    # + poisson.rvs(const, size = self.m*self.n)
        self.R = R
        self.C = C
        self.A_alpha = A_alpha
        self.true_row_labels_X = np.argmax(R, 1) + 1      #[x+1 for x in np.argmax(R_sort, axis =1).tolist()]
        self.true_column_labels_X = np.argmax(C, 1) + 1
        self.reorganized_matrix = R_sort@A_alpha@C_sort.T
        self.rowcluster_matrix = R_sort@R_sort.T@X_datamatrix
        self.columncluster_matrix = X_datamatrix@C_sort@C_sort.T

    def GenerateSELBM(self, alpha, model):                #theta = beta and alpha   M ?  idenfibility  
        
        R = np.zeros((self.m, self.g),dtype=int)
        C = np.zeros((self.n, self.s),dtype=int)
        E_mn = np.ones((self.m, self.n))
 
        for i,j in zip_longest(range(self.m), range(self.n)):
            if i is not None:
                R[i,:] = multinomial.rvs(1, self.pi, random_state=None)
            if j is not None:
                C[j,:] = multinomial.rvs(1, self.rho, random_state=None)
        R_sort = np.zeros_like(R)
        R_sort[np.arange(len(R)),np.sort(np.argmax(R,axis=1))] = 1
        C_sort = np.zeros_like(C)
        C_sort[np.arange(len(C)),np.sort(np.argmax(C,axis=1))] = 1
        X_datamatrix = np.zeros((self.m,self.n))
        E_noise = np.zeros((self.m,self.n))
        A_alpha = self.A_alpha
        if (model == "Poisson"):
            const = 2
            A_alpha = np.log(alpha)
            X_datamatrix = R@A_alpha@C.T
            beta_hat =  X_datamatrix@E_mn.T@X_datamatrix
            E_noise = np.reshape(poisson.rvs(const, size = self.m*self.n),(self.m, self.n))
        elif (model == "Normal"):
            const = 2
            sigma = 10^2        
            A_alpha = alpha/sigma             # mu/sigma^2
            X_datamatrix = R@A_alpha@C.T
            beta_hat =  E_mn
            E_noise = np.reshape(norm.rvs(loc = const, scale = 1, size = self.m*self.n),(self.m, self.n))
        elif (model == "Bernoulli"):  
            const = 0.7
            A_alpha = np.log(alpha/(1-alpha))
            X_datamatrix = R@A_alpha@C.T   
            beta_hat =  E_mn 
            E_noise = np.reshape(bernoulli.rvs(const, size = self.m*self.n),(self.m, self.n))                    
        elif (model == "Beta"): 
            const = 0.6
            A_alpha = alpha
            X_datamatrix = R@A_alpha@C.T
            beta_hat =  E_mn
            E_noise = np.reshape(beta.rvs(const, 1,  size = self.m*self.n),(self.m, self.n))
        else:
            print("Model name not found")
        self.X_datamatrix = X_datamatrix + E_noise
        self.R = R
        self.C = C
        self.A_alpha = A_alpha
        self.true_row_labels_X = np.argmax(R, 1) + 1      
        self.true_column_labels_X = np.argmax(C, 1) + 1
        self.reorganized_matrix = R_sort@A_alpha@C_sort.T
        self.rowcluster_matrix = R_sort@R_sort.T@X_datamatrix
        self.columncluster_matrix = X_datamatrix@C_sort@C_sort.T