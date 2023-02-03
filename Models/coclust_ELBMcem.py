# -*- coding: utf-8 -*-

"""
ELBMcem
"""

# Author: Saeid Hoseinipour <saeidhoseinipour9@gmail.com>
#                           <saeidhoseinipour@aut.ac.ir>

# License: 

import itertools
from math import *
from scipy.io import loadmat, savemat
import sys
import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state, check_array
#from coclust.utils.initialization import (random_init, check_numbers,check_array)
# use sklearn instead FR 08-05-19
from ..initialization import random_init
from ..io.input_checking import check_positive
import timeit


# from pylab import *


class CoclustELBMcem:
    """Clustering.
    Parameters
    ----------
    n_row_clusters : int, optional, default: 2
        Number of clusters to form
    n_col_clusters : int, optional, default: 2
        Number of clusters to form
    init : numpy array or scipy sparse matrix, \
        shape (n_features, n_clusters), optional, default: None
        Initial column or row labels
    max_iter : int, optional, default: 100
        Maximum number of iterations
    n_init : int, optional, default: 1
        Number of time the algorithm will be run with different
        initializations. The final results will be the best output of `n_init`
        consecutive runs.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    tol : float, default: 1e-9
        Relative tolerance with regards to criterion to declare convergence
    Model : str, default: "Poisson"     ("binary", "Poisson", "Normal", "Beta") 
        

    Attributes
    ----------
    row_labels_ : array-like, shape (n_rows,)
        cluster label of each row
    column_labels_ : array-like, shape (n_cols,)
        Bicluster label of each column
    criterion : float
        criterion obtained from the best run
    criterions : list of floats
        sequence of criterion values during the best run
    """

    def __init__(self, n_row_clusters = None, n_col_clusters = None, model = "Poisson" , init=None,
                 max_iter=100, n_init=1, tol=1e-9, random_state=None):
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters=n_col_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = check_random_state(random_state)
        self.row_labels_ =None
        self.column_labels_=None
        self.R = None
        self.C = None
        self.A_alpha = None
        self.criterions = []
        self.criterion = -np.inf
        self.model = model    # Model = ("binary", "Poisson", "Normal", "Beta") 
        self.runtime = None

    def fit(self, X, y=None):
        """Perform clustering.
        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """

        check_array(X, accept_sparse=True, dtype="numeric", order=None,
                    copy=False, force_all_finite=True, ensure_2d=True,
                    allow_nd=False, ensure_min_samples=self.n_row_clusters,
                    ensure_min_features=self.n_col_clusters, estimator=None)

        #check_positive(X)

        criterion = self.criterion
        criterions = self.criterions
        row_labels_ = self.row_labels_
        column_labels_ = self.column_labels_
        #runtime = self.runtime
        A_alpha = self.A_alpha
        R = self.R
        C = self.C
        runtime = []

        start = timeit.default_timer()

        X = sp.csr_matrix(X)
        #print(X)
        #X = X.astype(float)
        X=X.astype(int)       
        #print(X)

        random_state = check_random_state(self.random_state) 
        seeds = random_state.randint(np.iinfo(np.int32).max, size = self.n_init)
        print(seeds)
        for seed in seeds:
            self._fit_single(X, seed, y)
            if np.isnan(self.criterion):
                raise ValueError("matrix may contain negative or unexpected NaN values")
            # remember attributes corresponding to the best criterion
            if (self.criterion > criterion): 
                criterion = self.criterion
                criterions = self.criterions
                row_labels_ = self.row_labels_
                column_labels_ = self.column_labels_
                A_alpha = self.A_alpha
                R = self.R
                C = self.C
              
        self.random_state = random_state
        stop = timeit.default_timer()
        runtime.append(stop - start)
        # update attributes
        self.criterion = criterion
        self.criterions = criterions
        self.row_labels_ = row_labels_ 
        self.column_labels_ = column_labels_ 
        self.R = R
        self.C = C
        self.A_alpha = A_alpha        
        self.runtime = np.array(runtime).flatten()
        
    def _fit_single(self, X, random_state, y=None) :
        # X=X.astype(int)

        m, n = X.shape        
        g = self.n_row_clusters   #  g = number of row cluster
        s = self.n_col_clusters   #  s = number of column cluster

      
        E_mn = np.ones((m, n))  
        ############################################### S and beta 
        if (self.model == "Poisson"):
           beta = X@E_mn.T@X
           S = sp.lil_matrix(X)
           beta = sp.lil_matrix(beta)
        elif (self.model == "Normal"):
           beta = E_mn
           S = sp.lil_matrix(X)
        elif (self.model == "Bernoulli"):
           beta = E_mn
           S = sp.lil_matrix(X)
        elif (self.model == "Beta"):
           beta = E_mn
           S = np.log(X.data)
        else:
            print("Model name not found")

        N = X.sum()
        M = beta.sum()
        #print("M:---->"+str(M))
        const = 1./(1.*N*N)                                 # Safety parameter to avoid log(0) and  division by zero
        #print("const"+str(N))

        ###################################################### init of row labels
        if self.init is None:
            R = random_init(g, X.shape[0], random_state)   
        else:
            R = np.matrix(self.init, dtype=float)

        R = sp.lil_matrix(R)                                   # Random_init function returns a (2d)nd_array

        ##################################################### init of column labels
        if self.init is None:
            C = random_init(s, X.shape[1], random_state)
        else:
            C = np.matrix(self.init, dtype=float)

        C = sp.lil_matrix(C)

        #initial pi_k row proportion 
        n_k = R.sum()       
        pi_k = R.sum(axis=0)                               # r_{.k}
        pi_k = pi_k/n_k
        pi_k = np.asarray(pi_k)                 # (1,g)                             

        #initial rho_h column proportions 
        n_h = C.sum()
        #print(n_h)                                      
        rho_h = C.sum(axis=0)                               #  c_{.h}
        rho_h = rho_h/n_h
        rho_h = np.asarray(rho_h)             # (1,s)
        
        ################################################################## A_alpha and F_alpha               

        SS = R.T@S.multiply(beta)@C                 
        SS_n = R.T@beta@C

        D_A_F = np.nan_to_num(SS/SS_n+const)

        if (self.model == "Poisson"):
           A_alpha = sp.csr_matrix(D_A_F)
           A_alpha.data = np.log((A_alpha).data)
           F_alpha = sp.csr_matrix(D_A_F)
        elif (self.model == "Normal"):
           A_alpha = sp.csr_matrix(D_A_F)
           F_alpha = sp.csr_matrix(D_A_F)**2
        elif (self.model == "Bernoulli"):
           A_alpha = sp.csr_matrix(D_A_F)
           A_alpha.data = np.log((A_alpha).data/(1-(A_alpha).data))           
           F_alpha = sp.csr_matrix(D_A_F)
           F_alpha.data = -np.log(1-F_alpha.data)
        elif (self.model == "Beta"):
           A_alpha = sp.csr_matrix(D_A_F)
           F_alpha = sp.csr_matrix(D_A_F)
           F_alpha = np.array(1/F_alpha.data)
        else:
            print("Model name not found")


        ##########################################   Loop ##############################
        change = True
        c_init = float(-np.inf)
        c_list = []
        iteration = 0

        while change :
            change = False
            
            ################################################################### Rows

            ### CE step

            RA = A_alpha@(S.multiply(beta)@C).T             
            RF = F_alpha@(beta@C).T 
            Pi = np.vstack([np.log(pi_k+const)] * m)

            R1 = RA - RF  + Pi.T                               

            R1 = sp.csr_matrix(R1)

            R = sp.lil_matrix((R1.shape[1],g))
            R[np.arange(R1.shape[1]), R1.argmax(0).A1] = 1              

            R_cluster = sp.lil_matrix((R1.shape[1],g))
            R_cluster[np.arange(R1.shape[1]), np.sort(R1.argmax(axis = 0).A1)] = 1

            ### M step
            ### proportions of rows
            n_k = R.sum()
            pi_k = R.sum(axis=0)
            pi_k = pi_k/n
            pi_k = np.asarray(pi_k)
            print("pi_hat"+str(pi_k))
            #### parameters A_alpha and F_alpha

            SS = R.T@S.multiply(beta)@C          
            SS_n = R.T@beta@C

            D_A_F = np.nan_to_num(SS/SS_n+const)

            if (self.model == "Poisson"):
               A_alpha = sp.csr_matrix(D_A_F)
               A_alpha.data = np.log((A_alpha).data)
               F_alpha = sp.csr_matrix(D_A_F)
            elif (self.model == "Normal"):
               A_alpha = sp.csr_matrix(D_A_F)
               F_alpha = sp.csr_matrix(D_A_F)**2
            elif (self.model == "Bernoulli"):
               A_alpha = sp.csr_matrix(D_A_F)
               A_alpha.data = np.log((A_alpha).data/(1-(A_alpha).data))           
               F_alpha = sp.csr_matrix(D_A_F)
               F_alpha.data = -np.log(1-F_alpha.data)
            elif (self.model == "Beta"):
               A_alpha = sp.csr_matrix(D_A_F)
               F_alpha = sp.csr_matrix(D_A_F)
               F_alpha = np.array(1/F_alpha.data)
            else:
                print("Model name not found")

            # !!! A_alpha has been transformed to a (non-subscriptable)
            # COO matrix. Convert it back to CSR  FR 08-05-19
            #A_alpha = A_alpha.tocsr()
            
            #####avoid zero in A_alpha matrix
            
            #minval = np.min(A_alpha[np.nonzero(A_alpha)]) 
            #A_alpha[A_alpha == 0] = minval*0.00000001

            
            ##################################################################### Columns
           
            ### CE step

            CA = A_alpha.T@R.T@S.multiply(beta)                                 
            CF = F_alpha.T@R.T@beta
            RHO = np.vstack([np.log(rho_h+const)] * n)

            C1 = CA - CF + RHO.T    

            C1 = sp.csr_matrix(C1)

            C = sp.lil_matrix((C1.shape[1],s))
            C[np.arange(C1.shape[1]), C1.argmax(axis = 0).A1] = 1              

            C_cluster = sp.lil_matrix((C1.shape[1],s))
            C_cluster[np.arange(C1.shape[1]), np.sort(C1.argmax(axis = 0).A1)] = 1


            ### M step
            # proportions
            n_h = C.sum()
            rho_h = C.sum(axis=0)
            rho_h = rho_h/n_h
            rho_h = np.asarray(rho_h)
            print("rho_hat"+str(rho_h))

            ######################################## A_alpha and F_alpha        
            SS = R.T@S.multiply(beta)@C                
            SS_n = R.T@beta@C

            D_A_F = np.nan_to_num(SS/SS_n+const)
            print("alpha_hat"+str(D_A_F))
            if (self.model == "Poisson"):
               A_alpha = sp.csr_matrix(D_A_F)
               A_alpha.data = np.log((A_alpha).data)
               F_alpha = sp.csr_matrix(D_A_F)
               print(A_alpha,F_alpha)
            elif (self.model == "Normal"):
               A_alpha = sp.csr_matrix(D_A_F)
               F_alpha = sp.csr_matrix(D_A_F)**2
            elif (self.model == "Bernoulli"):
               A_alpha = sp.csr_matrix(D_A_F)
               A_alpha.data = np.nan_to_num(np.log((A_alpha).data/(1-(A_alpha).data)) , posinf = 0)
               F_alpha = sp.csr_matrix(D_A_F)
               F_alpha.data = -np.nan_to_num(np.log(1-F_alpha.data), posinf = 0)
               #print(A_alpha,F_alpha,F_alpha.data)
            elif (self.model == "Beta"):
               A_alpha = sp.csr_matrix(D_A_F)
               F_alpha = sp.csr_matrix(D_A_F)
               F_alpha = np.array(1/F_alpha.data)
            else:
                print("Model name not found")


            #minval=np.min(A_alpha[np.nonzero(A_alpha)]) 
            #A_alpha[A_alpha == 0] = minval*0.00000001
            #print(SS_n.T@F_alpha)

            ################################################################  Criterion (Complete log-likelihood)
            Ad = (A_alpha*SS).sum() - (SS_n*F_alpha).sum()
            #print("Ad"+str(Ad))
            #T1 = SS.T@A_alpha
            #print(np.diag(SS_n.T@F_alpha))
            #tr = np.trace(T1) # - np.trace(T2)
            sum_k = R.sum(axis=0)@np.log(pi_k+const).T 
            sum_h = C.sum(axis=0)@np.log(rho_h+const).T
            #print("sum"+str(sum_k)+str(sum_h))
            c =  sum_k + sum_h + Ad
            print("Log-likelihood=====>"+str(c))
                    
            iteration += 1
            if (np.abs(c - c_init)  > self.tol and iteration < self.max_iter): 
                c_init=c
                change=True
                c_list.append(c)

        self.max_iter = iteration
        self.criterion = c
        self.criterions = c_list
        self.R = R
        self.C = C
        self.A_alpha = A_alpha
        self.row_labels_ = [x+1 for x in R.toarray().argmax(axis=1).tolist()]
        self.column_labels_ = [x+1 for x in C.toarray().argmax(axis=1).tolist()]
        self.rowcluster_matrix = R_cluster.toarray()@R_cluster.toarray().T@X.toarray()
        self.columncluster_matrix = X.toarray()@C_cluster.toarray()@C_cluster.toarray().T
        self.reorganized_matrix = R_cluster.toarray()@R_cluster.toarray().T@X.toarray()@C_cluster.toarray()@C_cluster.toarray().T 
        self.reorganized_matrix_2 = R_cluster.toarray()@A_alpha@C_cluster.toarray().T              
