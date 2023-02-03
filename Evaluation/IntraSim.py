
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Intra-cluster Average Similarity (IAS): 
# measures the average similarity among documents belonging to the same cluster.

def IntraSim(cm, X):
	n = np.sum(cm, axis=1)
	#print(n[0])
	sim = cosine_similarity(X, X)
	IAS = []
	for k in range(cm.shape[0]):
		for i in range(n[k]):
			b = 0.5*(n[k]-1)
			#print(b)
			for j in range(i+1, n[k]):
				IAS.append(sim[i,j]/b)
				#print(IAS)
				pass
			pass
		pass
	return np.sum(IAS)/X.shape[0]		


def IntraCen(model, X):
	g = model.n_row_clusters
	#print(g)
	b = 0.5*g*(g-1)
	#print(b)
	sim_centroid = cosine_similarity(model.R.T@X, model.R.T@X)
	ICAS = []
	for k in range(g):
		for k_ in range(k+1, g):
			ICAS.append(sim_centroid[k,k_]/b)
			#print(ICAS)
			pass
		pass
	return np.sum(ICAS)