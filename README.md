# ELBMcoclust and SELBMcoclust
[![License](https://img.shields.io/badge/license-Apache%202-blue)](LICENSE)
![https://github.com/Saeidhoseinipour/NMTFcoclust](https://badgen.net/badge/ELBM/Coclust/black?icon=instgrame)

Sparse and Non-Sparse Exponential Family Latent Block Model for Co-clustering

The goal of the statistical approach is to analyze the behavior of the data by considering the probability distribution. The complete log-likelihood function for three version of LBM, Exponential LBM and Sparse Exponential LBM,  will be as follows:

-  **LBM**
```math
		L^{\text{LBM}}(\mathbf{r},\mathbf{c},\boldsymbol{\gamma})= \sum\limits_{i,k}r_{ik} \log\pi_{k} +\sum\limits_{j,h}  \log\rho_{h} c^{\top}_{jh}+
		\sum\limits_{i,j,k,h} r_{ik}\log \varphi(x_{ij};\alpha_{kh})c^{\top}_{hj}
```
-  **ELBM**
```math
	L^{\text{ELBM}}(\mathbf{r},\mathbf{c},\boldsymbol{\gamma})	\propto 
	\sum\limits_{k}
	r_{.k} \log\pi_{k} +	
	\sum\limits_{h}
	\log\rho_{h} c^{\top}_{h.} +
	\text{Tr}\left(
	(\mathbf{R}^{\top} (\mathbf{S_{x}}\odot \hat{\boldsymbol{\beta}}) \mathbf{C})^{\top}
	\mathbf{A}_{\boldsymbol{\alpha}}
	\right)
	- 
	\text{Tr}\left(
	(\mathbf{R}^{\top} (\mathbf{E}_{mn}\odot
	\hat{\boldsymbol{\beta}}) \mathbf{C})^{\top}
	\mathbf{F}_{\boldsymbol{\alpha}}
	\right)
```
-  **SELBM**
```math
\begin{align}
	L^{\text{SELBM}}(\mathbf{r},\mathbf{c},\boldsymbol{\gamma})
	\propto&
	\sum\limits_{k} r_{.k} \log\pi_{k} +	
	\sum\limits_{h}  c_{.h}\log\rho_{h}
	+
	\sum\limits_{k} 
	\left[
	\mathbf{R}^{\top}(\mathbf{S_{x}}\odot \hat{\boldsymbol{\beta}})\mathbf{C}	
	\right]_{kk}
	\left(
	A(\alpha_{kk}) - A(\alpha)
	\right)\nonumber\\
	&-  
	\sum\limits_{k}  
	[\mathbf{R}^{\top}	(\mathbf{E}_{mn} \odot \hat{\boldsymbol{\beta}} )\mathbf{C}]_{kk} 
	\left(
	F(A(\alpha_{kk})) -F(A(\alpha)) 
	\right) +N A(\alpha)	- 
	B F(A(\alpha)).
\end{align}
```

![](https://github.com/Saeidhoseinipour/ELBMcoclust/blob/main/Images/WebACE_SELBMvsELBM.png?raw=true)


## [Synthetic  Dataset](https://github.com/Saeidhoseinipour/ELBMcoclust/tree/main/Synthetic_Data)
![](https://github.com/Saeidhoseinipour/ELBMcoclust/blob/main/Images/Syn_data.png?raw=true)

## [Datasets](https://github.com/Saeidhoseinipour/ELBMcoclust/tree/main/Datasets)

| Datasets | Documents | Words | Sporsity | Number of clusters |
| -- | ----------- | -- | -- | -- |
| [CSTR](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/cstr.mat) | 475 | 1000 | 96% | 4 |
| [WebACE](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/WebACE..mat) |2340  |1000  | 91.83% |20  |
| [Classic3](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/classic3.mat) |3891  |4303  |98%  |3  |

## [Models](https://github.com/Saeidhoseinipour/ELBMcoclust/tree/main/Models)
```python
from ELBMcoclust.Models.coclust_ELBMcem import CoclustELBMcem
from ELBMcoclust.Models.coclust_SELBMcem import CoclustSELBMcem
```
```python
from NMTFcoclust.Evaluation.EV import Process_EV

ELBM = CoclustELBMcem(n_row_clusters = 4, n_col_clusters = 4, model = "Poisson")
ELBM.fit(X_CSTR)

SELBM = CoclustSELBMcem(n_row_clusters = 4, n_col_clusters = 4, model = "Poisson")
SELBM.fit(X_CSTR)

Process_Ev = Process_EV(true_labels ,X_CSTR, ELBM) 
```

```python
from sklearn.metrics import confusion_matrix 

confusion_matrix(true_labels, np.sort(ELBM.row_labels_))


array([[101,   0,   0,   0],
       [ 25,  46,   0,   0],
       [  0,   0,  68, 110],
       [  0,   0,   0, 125]], dtype=int64)
```


## Visualization

```python
from ELBMcoclust.Visualization.All_VS import All_Visualization

VS = All_Visualization(do_plot=True, save=True, dpi = 200)
a = 'Boxplot_Classic3_Final'
VS.boxplot_ELBM_SELBM('Classic3', a)
```

<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/ELBMcoclust/blob/main/Images/CSTR-1.png?raw=true" width="45%">

<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/ELBMcoclust/blob/main/Images/Classic3-1.png?raw=true" width="45%">

## Word cloud of SELBM for Classic3

<!-- 
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/ELBMcoclust/blob/main/Results/SELBM_Classic3.gif" width="45%">
 --> 
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/ELBMcoclust/blob/main/Images/Wordcloud_Classic3_SELBM.png?raw=true" width="80%">

<!--
![WC_ELBMcoclust](https://github.com/Saeidhoseinipour/ELBMcoclust/assets/43203342/5f33f01d-2236-4a47-9cf0-cf904fd047c3)
-->

## Cite
Please cite the following paper in your publication if you are using [ELBMcoclust]() in your research:

```bibtex
 @article{ELBMcoclust, 
    title={Sparse expoential family latent block model for co-clustering}, 
    DOI={Preprint}, 
    journal={Stat (preprint)}, 
    author={Saeid Hoseinipour, Mina Aminghafari, Adel Mohammadpour}, 
    year={2023}
} 
```
## References


[1] [Govaert and Nadif, Clustering with block mixture models, Pattern Recognition (2013).](https://www.sciencedirect.com/science/article/abs/pii/S0031320302000742)

[2] [Govaert and Nadif, Block clustering with Bernoulli mixture models: Comparison of different approaches, Computational Statistics and Data Analysis (2008).](https://www.sciencedirect.com/science/article/abs/pii/S0167947307003441)

[3] [Rodolphe Priam et al, Topographic Bernoulli block mixture mapping for binary tables, Pattern Analysis and Applications (2014).](https://link.springer.com/article/10.1007/s10044-014-0368-8)

[4] [Ailem, Melissa et al, Sparse Poisson latent block model for document clustering, IEEE Transactions on Knowledge and Data Engineering (2017).](https://ieeexplore.ieee.org/abstract/document/7876732) 

[5] [Saeid, Hoseinipour et al, Orthogonal Parametric Non-negative Matrix Tri-Factorization with $\alpha$-Divergence for Co-clustering](https://doi.org/10.1016/j.eswa.2023.120680)
