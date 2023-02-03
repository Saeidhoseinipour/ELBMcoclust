import argparse
import logging
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score  
from sklearn.metrics import adjusted_rand_score 
from ..Evaluation.IntraSim import IntraCen
from sklearn.metrics import confusion_matrix 
from sklearn.metrics.pairwise import cosine_similarity


from ..Models.coclust_ELBMcem import CoclustELBMcem
#from ..Models.coclust_SELBMcem import CoclustSELBMcem
from ..Models.coclust_SELBMcem_v6_2 import CoclustSELBMcem


class All_EV(object):
	"""docstring for All_EV_R (All EValuation Real datasets) """
	def __init__(self, X):
		super(All_EV, self).__init__()
		self.X = X
		self.Acc_row_ELBM = None
		self.ARI_row_ELBM = None
		self.ICAS_ELBM = None
		self.Acc_row_SELBM = None
		self.ARI_row_SELBM = None
		self.ICAS_SELBM = None
		self.Criterions_ELBM = None
		self.Criterions_SELBM = None


	def IntraCen(self, object_model):  
		g = object_model.n_row_clusters
		#s = object_model.n_col_clusters
		#print(g)
		b_g = 0.5*g*(g-1)
		#b_s = 0.5*s*(s-1)
		#print(b)
		sim_centroid_r = cosine_similarity(object_model.R.T@self.X, object_model.R.T@self.X)
		#sim_centroid_c = cosine_similarity(self.X@object_model.C, self.X@object_model.C)
		ICAS_r = []
		#ICAS_c = []
		for k in range(g):
			for k_ in range(k+1, g):
				ICAS_r.append(sim_centroid_r[k,k_]/b_g)
				#print(ICAS)
				pass
			pass
		return np.sum(ICAS_r)

	def Process_EV(self, true_labels, model):

		Acc_row_ELBM = []
		ARI_row_ELBM = []

		ICAS_ELBM = []
		runtime_ELBM = []

		Acc_row_SELBM = []
		ARI_row_SELBM = []

		ICAS_SELBM = []

		Criterions_ELBM = []
		Criterions_SELBM = []

		runtime_SELBM = []
		for i in np.arange(100):

			if (model == "Poisson"):
				ELBM = CoclustELBMcem(n_row_clusters = 20, n_col_clusters = 20, model = "Poisson", max_iter=1)
				SELBM = CoclustSELBMcem(n_row_clusters = 20, n_col_clusters = 20, model = "Poisson", max_iter=1) 
				ELBM.fit(self.X)
				SELBM.fit(self.X)
			elif (model == "Normal"):
				ELBM = CoclustELBMcem(n_row_clusters = 3, n_col_clusters = 3, model = "Normal", max_iter=1)
				SELBM = CoclustSELBMcem(n_row_clusters = 3, n_col_clusters = 3, model = "Normal", max_iter=1)
				ELBM.fit(self.X)
				SELBM.fit(self.X)
			elif (model == "Bernoulli"):
				ELBM = CoclustELBMcem(n_row_clusters = 3, n_col_clusters = 3, model = "Bernoulli", max_iter=1)
				SELBM = CoclustSELBMcem(n_row_clusters = 3, n_col_clusters = 3, model = "Bernoulli", max_iter=1)
				ELBM.fit(self.X)
				SELBM.fit(self.X)
			elif (model == "Gamma"):
				ELBM = CoclustELBMcem(n_row_clusters = 3, n_col_clusters = 3, model = "Beta", max_iter=1)
				SELBM = CoclustSELBMcem(n_row_clusters = 3, n_col_clusters = 3, model = "Beta", max_iter=1)
				ELBM.fit(self.X)
				SELBM.fit(self.X)
			else:
				print("Model name not found")

			ari_row_ELBM = adjusted_rand_score(np.sort(true_labels), np.sort(ELBM.row_labels_))
			acc_row_ELBM = accuracy_score(np.sort(true_labels), np.sort(ELBM.row_labels_))

			ari_row_SELBM = adjusted_rand_score(np.sort(true_labels), np.sort(SELBM.row_labels_))
			acc_row_SELBM = accuracy_score(np.sort(true_labels), np.sort(SELBM.row_labels_))

			icas_ELBM = self.IntraCen(ELBM)
			icas_SELBM = self.IntraCen(SELBM)

			Acc_row_ELBM.append(acc_row_ELBM)
			ARI_row_ELBM.append(ari_row_ELBM)

			Acc_row_SELBM.append(acc_row_SELBM)
			ARI_row_SELBM.append(ari_row_SELBM)

			ICAS_ELBM.append(icas_ELBM)
			ICAS_SELBM.append(icas_SELBM)

			Criterions_ELBM.append(ELBM.criterion[0])	
			Criterions_SELBM.append(SELBM.criterion[0])		

			print(confusion_matrix(np.sort(true_labels), np.sort(ELBM.row_labels_)))
			print(confusion_matrix(np.sort(true_labels), np.sort(SELBM.row_labels_)))
			runtime_ELBM.append(str(ELBM.runtime[0]))
			runtime_SELBM.append(str(SELBM.runtime[0]))

		return   [Acc_row_ELBM, Acc_row_SELBM, ARI_row_ELBM, ARI_row_SELBM, ICAS_ELBM, ICAS_SELBM, runtime_ELBM, runtime_SELBM, Criterions_ELBM, Criterions_SELBM]