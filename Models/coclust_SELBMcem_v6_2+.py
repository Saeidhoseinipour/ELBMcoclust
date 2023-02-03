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
import scipy.special as sc


# from pylab import *


class CoclustSELBMcem:
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
    model : str, default: "Poisson"     
        The name of distubtion based on (Sparse)Exponential Family Latent Block Model shuch as:
        "Poisson", "Bernoulli", "Normal", "Gamma", "Beta".
        

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
        self.row_labels_ = None
        self.column_labels_= None
        self.A_star = None
        self.F_star = None
        self.R = None
        self.C = None
        self.reorganized_matrix = None
        self.reorganized_matrix_2 = None
        self.rowcluster_matrix = None
        self.columncluster_matrix = None
        self.R_cluster = None
        self.C_cluster = None
        self.D_A_F_alpha = None
        self.D_A_F_alpha_kk = None
        self.criterions = []
        self.criterion = -np.inf
        self.model = model    # model = ("Bernoulli", "Poisson", "Normal", "Beta") 
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
        A_star = self.A_star 
        F_star = self.F_star 
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
                A_star = self.A_star 
                F_star = self.F_star 
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
        self.runtime = runtime
        self.A_star = A_star
        self.F_star = F_star 
        self.R = R
        self.C = C


                
        
    def _fit_single(self, X, random_state, y=None) :
        # X=X.astype(int)

        m, n = X.shape        
        g = self.n_row_clusters   #  g = number of row cluster
        s = self.n_col_clusters   #  s = number of column cluster

      
        E_mn = np.ones((m, n))  
        E_sg = np.ones((s, g))
        E_gg = np.ones((g, g))
        ############################################### S and beta 
        if (self.model == "Poisson"):
           beta = X@E_mn.T@X
           S = sp.lil_matrix(X)
           beta = sp.lil_matrix(beta)
        elif (self.model == "Normal"):
           beta = E_mn
           S = sp.lil_matrix(X)
           beta = sp.lil_matrix(beta)
        elif (self.model == "Bernoulli"):
           beta = E_mn
           S = sp.lil_matrix(X)
           beta = sp.lil_matrix(beta)
        elif (self.model == "Beta"):
           beta = E_mn
           S = np.log(X.toarray())
           S = sp.lil_matrix(S)
           beta = sp.lil_matrix(beta)
        else:
            print("Model name not found")

        N = X.sum()
        const = 1./(1.*N*N)                                 # Safety parameter to avoid log(0) and  division by zero
        #print("const"+str(const))

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
        
        ################################################### (A_alpha_kk - A_alpha) and (F_alpha_kk - F_alpha)               

        SS = R.T@S.multiply(beta)@C                 
        SS_n = R.T@beta@C
        #print(SS.toarray(),SS_n.toarray())
        #print(SS,SS_n)
        SS_a = SS.sum()-np.diag(SS.toarray()).sum()                             ####v4
        SS_a_n = SS_n.sum()-np.diag(SS_n.toarray()).sum()                         ####v4

        D_A_F_alpha_kk = np.nan_to_num(SS.toarray()/SS_n.toarray())          #+ self.tol
        D_A_F_alpha_kk = np.diag(np.max(D_A_F_alpha_kk, axis=1))
        D_A_F_alpha = np.nan_to_num(SS_a/SS_a_n)          #+ self.tol


        if (self.model == "Bernoulli"):
            A_alpha_kk__A_alpha = np.nan_to_num(((1- D_A_F_alpha)*(D_A_F_alpha_kk))/((1-D_A_F_alpha_kk)*(D_A_F_alpha)))  #+ self.tol
            #A_alpha_kk__A_alpha[A_alpha_kk__A_alpha<0] = 1
            A_alpha_kk__A_alpha = np.log(A_alpha_kk__A_alpha)                 # values between 0 to 1 is negative 
            print("Here")
            print(A_alpha_kk__A_alpha)
            A_alpha = np.nan_to_num(np.log(D_A_F_alpha/(1-D_A_F_alpha)))               # + self.tol
            print(A_alpha)
            F_alpha_kk__F_alpha = np.nan_to_num((1- D_A_F_alpha)/(1-D_A_F_alpha_kk))
            #F_alpha_kk__F_alpha[F_alpha_kk__F_alpha<0] = 1
            F_alpha_kk__F_alpha = np.log(F_alpha_kk__F_alpha)
            print(F_alpha_kk__F_alpha)
            F_alpha = -np.log((1-D_A_F_alpha))
        elif (self.model == "Normal"):
            A_alpha_kk__A_alpha = 0.5*np.nan_to_num((D_A_F_alpha_kk)-(D_A_F_alpha))
            F_alpha_kk__F_alpha = 0.5*np.nan_to_num((D_A_F_alpha_kk**2)-(D_A_F_alpha**2))
            A_alpha = 0.5*np.nan_to_num(D_A_F_alpha)
            F_alpha = 0.5*np.nan_to_num(D_A_F_alpha**2)
        elif (self.model == "Poisson"):
            A_alpha_kk__A_alpha = np.nan_to_num((D_A_F_alpha_kk)/(D_A_F_alpha))
            #print("Here_0")
            #print(A_alpha_kk__A_alpha)
            #A_alpha_kk__A_alpha[A_alpha_kk__A_alpha<0] = 0
            #A_alpha_kk__A_alpha = sp.csr_matrix(A_alpha_kk__A_alpha)
            A_alpha_kk__A_alpha = np.log(A_alpha_kk__A_alpha)
            #print(A_alpha_kk__A_alpha,np.shape(A_alpha_kk__A_alpha))

            A_alpha = np.nan_to_num(np.log(D_A_F_alpha))

            F_alpha_kk__F_alpha = np.nan_to_num((D_A_F_alpha_kk)-(D_A_F_alpha))
            #F_alpha_kk__F_alpha[F_alpha_kk__F_alpha<0] = 0
            #F_alpha_kk__F_alpha = sp.csr_matrix(F_alpha_kk__F_alpha)

            F_alpha = D_A_F_alpha
        elif (self.model == "Beta"):
            A_alpha_kk__A_alpha = np.nan_to_num((D_A_F_alpha_kk)-(D_A_F_alpha))
            F_alpha_kk__F_alpha = np.log(sc.beta(D_A_F_alpha_kk,1))-np.log(sc.beta(D_A_F_alpha,1))
            A_alpha = np.nan_to_num(D_A_F_alpha)
            F_alpha = np.log(sc.beta(D_A_F_alpha,1))
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
            T_1 = S.multiply(beta)@C
            #print(T_1.toarray())
            T_2 = beta@C
            #print(T_2.toarray())
            T_3 = S.multiply(beta)@C@E_sg
            #print(T_3)
            RAF = np.ones((m,np.min((g,s))))
            for i in np.arange(m):
                for k in np.arange(np.min((g,s))):
                    #print(A_alpha_kk__A_alpha[k])
                    #print((S.multiply(beta)@C).toarray()[i][k])
                    #print((beta@C).toarray()[i][k])
                    #print(A_alpha) 
                    #print((S.multiply(beta)@C@E_sg)[i][k])
                    #print(F_alpha_kk__F_alpha[k])
                    RAF[i][k] = (A_alpha_kk__A_alpha)[k]*(T_1.toarray())[i][k] - (F_alpha_kk__F_alpha)[k]*(T_2.toarray())[i][k] + A_alpha*(T_3)[i][k]

            Pi = np.vstack([np.log(pi_k + const)] * m)

            Pi = sp.lil_matrix(Pi)
            #print(RAF,np.shape(RAF))

            R1 = Pi + RAF          
            #print(Pi,RAF)
            R1 = sp.csr_matrix(R1)
            #print(R1.toarray())
            R = sp.lil_matrix((R1.shape[0],g))
            R[np.arange(R1.shape[0]), R1.argmax(1).A1] = 1
            print("R====>"+str(R.toarray()))
            R_cluster = sp.lil_matrix((R1.shape[0],g))
            R_cluster[np.arange(R1.shape[0]), np.sort(R1.argmax(axis = 1).A1)] = 1
            #print(R_cluster)
            ### M step
            ### proportions of rows
            n_k = R.sum()
            pi_k = R.sum(axis=0)
            pi_k = pi_k/n
            pi_k = np.asarray(pi_k)

            #### parameters A_alpha and F_alpha

            SS = R.T@S.multiply(beta)@C          
            SS_n = R.T@beta@C

            SS_a = SS.sum()-np.diag(SS.toarray()).sum()                             ####v3
            SS_a_n = SS_n.sum()-np.diag(SS_n.toarray()).sum()                         ####v3


            D_A_F_alpha_kk = np.nan_to_num(SS.toarray()/SS_n.toarray())
            D_A_F_alpha_kk = np.diag(np.max(D_A_F_alpha_kk, axis=1))

            D_A_F_alpha = np.nan_to_num(SS_a/SS_a_n)         # number

            if (self.model == "Bernoulli"):
                A_alpha_kk__A_alpha = np.nan_to_num(((1- D_A_F_alpha)*(D_A_F_alpha_kk))/((1-D_A_F_alpha_kk)*(D_A_F_alpha)))  #+ self.tol
                #A_alpha_kk__A_alpha[A_alpha_kk__A_alpha<0] = 1
                A_alpha_kk__A_alpha = np.log(A_alpha_kk__A_alpha)                 # values between 0 to 1 is negative 

                A_alpha = np.nan_to_num(np.log(D_A_F_alpha/(1-D_A_F_alpha)))               # + self.tol

                F_alpha_kk__F_alpha = np.nan_to_num((1- D_A_F_alpha)/(1-D_A_F_alpha_kk))
                #F_alpha_kk__F_alpha[F_alpha_kk__F_alpha<0] = 1
                F_alpha_kk__F_alpha = np.log(F_alpha_kk__F_alpha)

                F_alpha = -np.log((1-D_A_F_alpha))
            elif (self.model == "Normal"):
                A_alpha_kk__A_alpha = 0.5*np.nan_to_num((D_A_F_alpha_kk)-(D_A_F_alpha))
                F_alpha_kk__F_alpha = 0.5*np.nan_to_num((D_A_F_alpha_kk**2)-(D_A_F_alpha**2))
                A_alpha = 0.5*np.nan_to_num(D_A_F_alpha)
                F_alpha = 0.5*np.nan_to_num(D_A_F_alpha**2)
            elif (self.model == "Poisson"):
                #print("Here")
                #print(D_A_F_alpha_kk)
                #print(D_A_F_alpha)
                A_alpha_kk__A_alpha = np.nan_to_num((D_A_F_alpha_kk)/(D_A_F_alpha))
                #A_alpha_kk__A_alpha[A_alpha_kk__A_alpha<0] = 0
                #A_alpha_kk__A_alpha = sp.csr_matrix(A_alpha_kk__A_alpha)
                A_alpha_kk__A_alpha = np.nan_to_num(np.log(A_alpha_kk__A_alpha),posinf = 0)

                A_alpha = np.nan_to_num(np.log(D_A_F_alpha))

                F_alpha_kk__F_alpha = np.nan_to_num((D_A_F_alpha_kk)-(D_A_F_alpha))
                #F_alpha_kk__F_alpha[F_alpha_kk__F_alpha<0] = 0
                #F_alpha_kk__F_alpha = sp.csr_matrix(F_alpha_kk__F_alpha)
                
                F_alpha = D_A_F_alpha
            elif (self.model == "Beta"):
                A_alpha_kk__A_alpha = np.nan_to_num((D_A_F_alpha_kk)-(D_A_F_alpha))
                A_alpha_kk__A_alpha[A_alpha_kk__A_alpha<0] = 1
                F_alpha_kk__F_alpha = np.log(sc.beta(D_A_F_alpha_kk,1))-np.log(sc.beta(D_A_F_alpha,1))
                A_alpha = np.nan_to_num(D_A_F_alpha)
                F_alpha = np.log(sc.beta(D_A_F_alpha,1))
                print(A_alpha_kk__A_alpha,F_alpha_kk__F_alpha,F_alpha, A_alpha)                
            else:
                print("Model name not found")

            # !!! A_alpha has been transformed to a (non-subscriptable)
            # COO matrix. Convert it back to CSR  FR 08-05-19
            #A_alpha = A_alpha.tocsr()
            
            #####avoid zero in A_alpha matrix
            
            #minval = np.min(A_alpha_kk__A_alpha[np.nonzero(A_alpha_kk__A_alpha)]) 
            #A_alpha_kk__A_alpha[A_alpha_kk__A_alpha == 0] = minval*0.00000001

            
            ##################################################################### Columns
           
            ### CE step
            T_1 = R.T@S.multiply(beta)
            #print(T_1.toarray())
            T_2 = R.T@beta
            #print(T_2.toarray())
            T_3 = E_sg@R.T@S.multiply(beta)
            #print(T_3)

            CAF = np.ones((n,np.max((g,s))))
            for j in np.arange(n):
                for h in np.arange(np.max((g,s))):
                    if h <= np.min((g,s)):     
                        #print("Here_2")
                        #print(np.diag(A_alpha_kk__A_alpha))
                        #print(T_1.toarray())
                        #print(A_alpha_kk__A_alpha.T@T_1.toarray())
                        #print(F_alpha_kk__F_alpha.T@T_2.toarray())
                        #print(A_alpha*T_3)
                        CAF[j][h] = (np.diag(A_alpha_kk__A_alpha).T@T_1.toarray())[h][j] - (np.diag(F_alpha_kk__F_alpha).T@T_2.toarray())[h][j] + A_alpha*(T_3)[h][j]
                    else:
                        CAF[j][h] =  A_alpha*(T_3)[h][j]
            RHO = np.vstack([np.log(rho_h + const)] * n)
            #print(CAF,np.shape(CAF))
            #print(RHO)

            C1 = RHO + CAF
            #print(C1)
            C1 = sp.csr_matrix(C1)

            C = sp.lil_matrix((C1.shape[0],s))
            C[np.arange(C1.shape[0]), C1.argmax(axis = 1).A1] = 1              
            print("C====>"+str(C.toarray()))
            C_cluster = sp.lil_matrix((C1.shape[0],s))
            C_cluster[np.arange(C1.shape[0]), np.sort(C1.argmax(axis = 1).A1)] = 1


            ### M step
            # proportions
            n_h = C.sum()
            rho_h = C.sum(axis=0)
            rho_h = rho_h/n_h
            rho_h = np.asarray(rho_h)

            ######################################## A_alpha_kk__A_alpha and F_alpha_kk__F_alpha        

            SS = R.T@S.multiply(beta)@C 
            #print(SS.toarray())               
            SS_n = R.T@beta@C
            #print(SS_n)
            SS_a = SS.sum()-np.diag(SS.toarray()).sum()                             ####v3
            SS_a_n = SS_n.sum()-np.diag(SS_n.toarray()).sum()

            
            D_A_F_alpha_kk = np.nan_to_num(SS.toarray()/SS_n.toarray())
            D_A_F_alpha_kk = np.diag(np.max(D_A_F_alpha_kk, axis=1))
            #print(D_A_F_alpha_kk)
            #print(D_A_F_alpha)
            D_A_F_alpha = np.nan_to_num(SS_a/SS_a_n)
            #print("alpha_kk"+str(D_A_F_alpha_kk))
            #print("alpha"+str(D_A_F_alpha))
            if (self.model == "Bernoulli"):
                A_alpha_kk__A_alpha = np.nan_to_num(((1- D_A_F_alpha)*(D_A_F_alpha_kk))/((1-D_A_F_alpha_kk)*(D_A_F_alpha)))  #+ self.tol
                #A_alpha_kk__A_alpha[A_alpha_kk__A_alpha<0] = 1
                A_alpha_kk__A_alpha = np.log(A_alpha_kk__A_alpha)                 # values between 0 to 1 is negative 

                A_alpha = np.nan_to_num(np.log(D_A_F_alpha/(1-D_A_F_alpha)))               # + self.tol

                F_alpha_kk__F_alpha = np.nan_to_num((1- D_A_F_alpha)/(1-D_A_F_alpha_kk))
                #F_alpha_kk__F_alpha[F_alpha_kk__F_alpha<0] = 1
                F_alpha_kk__F_alpha = np.log(F_alpha_kk__F_alpha)

                F_alpha = -np.log((1-D_A_F_alpha))
            elif (self.model == "Normal"):
                sigma = 1
                sigma_kk = np.ones((1, np.min(s,g)))
                A_alpha_kk__A_alpha = 0.5*np.nan_to_num((D_A_F_alpha_kk)-(D_A_F_alpha))
                F_alpha_kk__F_alpha = 0.5*np.nan_to_num((D_A_F_alpha_kk**2)-(D_A_F_alpha**2))
                A_alpha = 0.5*np.nan_to_num(D_A_F_alpha)
                F_alpha = 0.5*np.nan_to_num(D_A_F_alpha**2)
            elif (self.model == "Poisson"):
                A_alpha_kk__A_alpha = np.nan_to_num((D_A_F_alpha_kk)/(D_A_F_alpha), posinf = 0)
                #print(A_alpha_kk__A_alpha)
                A_alpha_kk__A_alpha[A_alpha_kk__A_alpha == 0] = 1
                #A_alpha_kk__A_alpha = sp.csr_matrix(A_alpha_kk__A_alpha)
                A_alpha_kk__A_alpha = np.log(A_alpha_kk__A_alpha)
                #print(A_alpha_kk__A_alpha)
                A_alpha = np.nan_to_num(np.log(D_A_F_alpha))

                F_alpha_kk__F_alpha = np.nan_to_num((D_A_F_alpha_kk)-(D_A_F_alpha))
                #F_alpha_kk__F_alpha[F_alpha_kk__F_alpha<0] = 0
                #F_alpha_kk__F_alpha = sp.csr_matrix(F_alpha_kk__F_alpha)
                #FA_alpha = np.nan_to_num(D_A_F_alpha)
                
                F_alpha = D_A_F_alpha
            elif (self.model == "Beta"):
                A_alpha_kk__A_alpha = np.nan_to_num((D_A_F_alpha_kk)-(D_A_F_alpha))
                F_alpha_kk__F_alpha = np.log(sc.beta(D_A_F_alpha_kk,1))-np.log(sc.beta(D_A_F_alpha,1))
                A_alpha = np.nan_to_num(D_A_F_alpha)
                F_alpha = np.log(sc.beta(D_A_F_alpha,1))
            else:
                print("Model name not found")

            #minval=np.min(A_alpha[np.nonzero(A_alpha)]) 
            #A_alpha[A_alpha == 0] = minval*0.00000001

            ################################################################  Criterion (Complete log-likelihood)
            tr = []
            for k in np.arange(g):
                #print((A_alpha_kk__A_alpha)[k])
                #print((SS.toarray())[k][k])
                #print((F_alpha_kk__F_alpha)[k])
                #print((SS_n.toarray())[k][k])
                tr.append((A_alpha_kk__A_alpha)[k]*(SS.toarray())[k][k] - (F_alpha_kk__F_alpha)[k]*(SS_n.toarray())[k][k])

            sum_k = R.sum(axis=0)@np.log(pi_k+const).T 
            sum_h = C.sum(axis=0)@np.log(rho_h+const).T

            c =  sum_k + sum_h + np.sum(tr) 
            print("Log-likelihood=====>"+str(c))
                    
            iteration += 1
            if (np.abs(c - c_init)  > self.tol and iteration < self.max_iter): 
                c_init = c 
                change=True
                c_list.append(c)

        self.max_iter = iteration
        self.criterion = c
        self.criterions = c_list
        self.row_labels_ = [x+1 for x in R.toarray().argmax(axis=1).tolist()]
        #print(self.row_labels_)
        self.column_labels_ = [x+1 for x in C.toarray().argmax(axis=1).tolist()]
        #print(self.column_labels_)
        self.rowcluster_matrix = R_cluster.toarray()@R_cluster.toarray().T@X.toarray()
        self.columncluster_matrix = X.toarray()@C_cluster.toarray()@C_cluster.toarray().T
        self.reorganized_matrix = R_cluster.toarray()@R_cluster.toarray().T@X.toarray()@C_cluster.toarray()@C_cluster.toarray().T   
        self.reorganized_matrix_2 = R_cluster.toarray()@np.diag(np.max(R_cluster.toarray().T@X.toarray()@C_cluster.toarray(), axis = 1))@C_cluster.toarray().T                     
        #self.A_star = A_star
        #self.F_star = F_star
        self.R = R_cluster
        self.C = C_cluster