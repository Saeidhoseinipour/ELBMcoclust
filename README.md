# ELBMcoclust
Exponential Family Latent Block Model for Co-clustering

The goal of the statistical approach is to analyze the behavior of the data by considering the probability distribution, 
while the goal of the linear algebra approach is to handle the data using matrices and tensors.

-  **LBM**
```math
		L^{LBM}(\mathbf{r},\mathbf{c},\boldsymbol{\gamma})= \sum\limits_{i,k}r_{ik} \log\pi_{k} +\sum\limits_{j,h}  \log\rho_{h} c^{\top}_{jh}+
		\sum\limits_{i,j,k,h} r_{ik}\log \varphi(x_{ij};\alpha_{kh})c^{\top}_{hj}
```
-  **ELBM**
```math
	L^{ELBM}(\mathbf{r},\mathbf{c},\boldsymbol{\gamma})	\propto 
	\sum\limits_{k}
	r_{.k} \log\pi_{k} +	
	\sum\limits_{h}
	\log\rho_{h} c^{\top}_{h.} +
	Tr\left(
	(\mathbf{R}^{\top} (\mathbf{S_{x}}\odot \hat{\boldsymbol{\beta}}) \mathbf{C})^{\top}
	\mathbf{A}_{\boldsymbol{\alpha}}
	\right)
	- 
	Tr\left(
	(\mathbf{R}^{\top} (\mathbf{E}_{mn}\odot
	\hat{\boldsymbol{\beta}}) \mathbf{C})^{\top}
	\mathbf{F}_{\boldsymbol{\alpha}}
	\right)
```
-  **SELBM**
```math
\begin{align*}
	L^{SELBM}(\boldsymbol{r},\boldsymbol{c},\boldsymbol{\gamma})
	\propto&
	\sum\limits_{k} r_{.k} \log\pi_{k} +	
	\sum\limits_{h}  c_{.h}\log\rho_{h} \nonumber\\
	&+
	\sum\limits_{k} 
	\left[
	\mathbf{R}^{\top}(\mathbf{S}\odot \hat{\boldsymbol{\beta}})\mathbf{C}	
	\right]_{kk}
	\left(
	A(\alpha_{kk}) - A(\alpha)
	\right)\nonumber\\
	&+
	N A(\alpha)\nonumber\\
	&-  
	\sum\limits_{k}  
	[\mathbf{R}^{\top}	(\mathbf{E}_{mn} \odot \hat{\boldsymbol{\beta}} )\mathbf{C}]_{kk} 
	\left(
	F(A(\alpha_{kk})) -F(A(\alpha)) 
	\right) \nonumber\\
	&- 
	B F(A(\alpha)) 
\end{align*}
```

![](https://github.com/Saeidhoseinipour/ELBMcoclust/blob/main/Images/WebACE_SELBM_Reorg.png?raw=true)

## [Datasets](https://github.com/Saeidhoseinipour/ELBMcoclust/tree/main/Datasets)

| Datasets | Documents | Words | Sporsity | Number of clusters |
| -- | ----------- | -- | -- | -- |
| [CSTR](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/cstr.mat) | 475 | 1000 | 96% | 4 |
| [WebACE](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/WebACE..mat) |2340  |1000  | 91.83% |20  |
| [Classic3](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/classic3.mat) |3891  |4303  |98%  |3  |

## Models
```python
from ..Models.ELBMcem import 
```
