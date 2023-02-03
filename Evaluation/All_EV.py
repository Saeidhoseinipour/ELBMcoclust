import argparse
import logging
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score  
from sklearn.metrics import adjusted_rand_score 
from sklearn.metrics.cluster import normalized_mutual_info_score 
from ..Evaluation.IntraSim import IntraCen
from sklearn.metrics import confusion_matrix 
from sklearn.metrics.pairwise import cosine_similarity


from ..Models.coclust_ELBMcem import CoclustELBMcem
#from ..Models.coclust_ELBMcem_v2 import CoclustELBMcem
#from ..Models.coclust_SELBMcem import CoclustSELBMcem
#from ..Models.coclust_SELBMcem_v2 import CoclustSELBMcem
from ..Models.coclust_SELBMcem_v6_2 import CoclustSELBMcem



class All_EV(object):
	"""docstring for All_EV"""
	def __init__(self, X):
		super(All_EV, self).__init__()
		self.X = X
		self.Acc_row_ELBM = None
		self.Acc_col_ELBM = None
		self.ARI_row_ELBM = None
		self.ARI_col_ELBM = None

		self.ICAS_ELBM = None


		self.Acc_row_SELBM = None
		self.Acc_col_SELBM = None
		self.ARI_row_SELBM = None
		self.ARI_col_SELBM = None
		self.ICAS_SELBM = None
		self.Criterions_ELBM = None
		self.Criterions_SELBM = None

	def IntraCen(self, object_model):  
		g = object_model.n_row_clusters
		s = object_model.n_col_clusters
		#print(g)
		b_g = 0.5*g*(g-1)
		b_s = 0.5*s*(s-1)
		#print(b)
		sim_centroid_r = cosine_similarity(object_model.R.T@self.X, object_model.R.T@self.X)
		sim_centroid_c = cosine_similarity(object_model.C.T@self.X.T, object_model.C.T@self.X.T)
		ICAS_r = []
		ICAS_c = []
		for k in range(g):
			for k_ in range(k+1, g):
				ICAS_r.append(sim_centroid_r[k,k_]/b_g)
				#print(ICAS)
				pass
			pass

		for h in range(s):
			for h_ in range(h+1, s):
				ICAS_c.append(sim_centroid_c[h,h_]/b_s)
				#print(ICAS)
				pass
			pass
		return [np.sum(ICAS_r), np.sum(ICAS_c)]

	def Process_EV(self, args, model):

		Acc_row_ELBM = []
		Acc_col_ELBM = []
		ARI_row_ELBM = []
		ARI_col_ELBM = []
		ICAS_row_ELBM = []
		ICAS_col_ELBM = []
		NMI_row_ELBM = []
		NMI_col_ELBM = []
		runtime_ELBM = []

		Acc_row_SELBM = []
		Acc_col_SELBM = []
		ARI_row_SELBM = []
		ARI_col_SELBM = []
		ICAS_row_SELBM = []
		ICAS_col_SELBM = []
		NMI_row_SELBM = []
		NMI_col_SELBM = []
		runtime_SELBM = []

		Criterions_ELBM = []
		Criterions_SELBM = []

		for i in np.arange(10):

			if (model == "Poisson"):
				ELBM = CoclustELBMcem(n_row_clusters = 3, n_col_clusters = 3, model = "Poisson", max_iter=100)
				SELBM = CoclustSELBMcem(n_row_clusters = 3, n_col_clusters = 3, model = "Poisson", max_iter=100) 
				ELBM.fit(self.X)
				SELBM.fit(self.X)
			elif (model == "Normal"):
				ELBM = CoclustELBMcem(n_row_clusters = 3, n_col_clusters = 3, model = "Normal", max_iter=100)
				SELBM = CoclustSELBMcem(n_row_clusters = 3, n_col_clusters = 3, model = "Normal", max_iter=100)
				ELBM.fit(self.X)
				SELBM.fit(self.X)
			elif (model == "Bernoulli"):
				ELBM = CoclustELBMcem(n_row_clusters = 3, n_col_clusters = 3, model = "Bernoulli", max_iter=100)
				SELBM = CoclustSELBMcem(n_row_clusters = 3, n_col_clusters = 3, model = "Bernoulli", max_iter=100)
				ELBM.fit(self.X)
				SELBM.fit(self.X)
			elif (model == "Gamma"):
				ELBM = CoclustELBMcem(n_row_clusters = 3, n_col_clusters = 3, model = "Beta", max_iter=100)
				SELBM = CoclustSELBMcem(n_row_clusters = 3, n_col_clusters = 3, model = "Beta", max_iter=100)
				ELBM.fit(self.X)
				SELBM.fit(self.X)
			else:
				print("Model name not found")

			ari_row_ELBM = adjusted_rand_score(np.sort(args.true_row_labels_X), np.sort(ELBM.row_labels_))
			acc_row_ELBM = accuracy_score(np.sort(args.true_row_labels_X), np.sort(ELBM.row_labels_))
			nmi_row_ELBM = normalized_mutual_info_score(np.sort(args.true_row_labels_X), np.sort(ELBM.row_labels_))

			ari_row_SELBM = adjusted_rand_score(np.sort(args.true_row_labels_X), np.sort(SELBM.row_labels_))
			acc_row_SELBM = accuracy_score(np.sort(args.true_row_labels_X), np.sort(SELBM.row_labels_))
			nmi_row_SELBM = normalized_mutual_info_score(np.sort(args.true_row_labels_X), np.sort(SELBM.row_labels_))

			icas_row_ELBM, icas_col_ELBM  = self.IntraCen(ELBM)
			icas_row_SELBM, icas_col_SELBM = self.IntraCen(SELBM)

			ari_col_ELBM = adjusted_rand_score(np.sort(args.true_column_labels_X), np.sort(ELBM.column_labels_))
			acc_col_ELBM = accuracy_score(np.sort(args.true_column_labels_X), np.sort(ELBM.column_labels_))
			nmi_col_ELBM = normalized_mutual_info_score(np.sort(args.true_row_labels_X), np.sort(ELBM.column_labels_))

			ari_col_SELBM = adjusted_rand_score(np.sort(args.true_column_labels_X), np.sort(SELBM.column_labels_))
			acc_col_SELBM = accuracy_score(np.sort(args.true_column_labels_X), np.sort(SELBM.column_labels_))
			nmi_col_SELBM = normalized_mutual_info_score(np.sort(args.true_row_labels_X), np.sort(ELBM.column_labels_))

			#print("Accuracy              ======>"             + str(acc_row) + str(acc_col))
			#print("Adjusted Rand Index   ======>"             + str(ari_col) + str(ari_col))
			#print("ICAS   ======>"             + str(icas))
			#print("Runtime               ======>"             + str(ELBM.runtime))
			#print(cm)
			Acc_row_ELBM.append(acc_row_ELBM)
			Acc_col_ELBM.append(acc_col_ELBM)
			ARI_row_ELBM.append(ari_row_ELBM)
			ARI_col_ELBM.append(ari_col_ELBM)
			NMI_row_ELBM.append(nmi_row_ELBM)
			NMI_col_ELBM.append(nmi_col_ELBM)

			Acc_row_SELBM.append(acc_row_SELBM)
			Acc_col_SELBM.append(acc_col_SELBM)
			ARI_row_SELBM.append(ari_row_SELBM)
			ARI_col_SELBM.append(ari_col_SELBM)	
			NMI_row_SELBM.append(nmi_row_SELBM)
			NMI_col_SELBM.append(nmi_col_SELBM)

			ICAS_row_ELBM.append(icas_row_ELBM)
			ICAS_row_SELBM.append(icas_row_SELBM)
			ICAS_col_ELBM.append(icas_col_ELBM)
			ICAS_col_SELBM.append(icas_col_SELBM)

			runtime_ELBM.append(str(ELBM.runtime)[1:-1])
			runtime_SELBM.append(str(SELBM.runtime)[1:-1])

			Criterions_ELBM.append(ELBM.criterion[0])	
			Criterions_SELBM.append(SELBM.criterion[0])


		return   [Acc_row_ELBM, Acc_col_ELBM, Acc_row_SELBM, Acc_col_SELBM, ARI_row_ELBM, ARI_col_ELBM, ARI_row_SELBM, ARI_col_SELBM, NMI_row_ELBM, NMI_col_ELBM, NMI_row_SELBM, NMI_col_SELBM, ICAS_row_ELBM, ICAS_col_ELBM, ICAS_col_SELBM,ICAS_col_SELBM, runtime_ELBM, runtime_SELBM, Criterions_ELBM, Criterions_SELBM]