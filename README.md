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

\begin{center}
	\begin{table}[H]
			\caption{Table of 
			the elements of complete log-likelihood for two case sparse.}\label{tab:EF}
		\centering
	\begin{tabular*}{300pt}{@{\extracolsep\fill}cccccc@{\extracolsep\fill}}%
		\toprule
	 \textbf{Distribution} & $\mathbf{S_{x}}$ & $\mathbf{A}_{\boldsymbol{\alpha}}$&$\mathbf{F}_{\boldsymbol{\alpha}}$ &$\hat{\boldsymbol{\beta}}$ &\textbf{$\varphi(x_{ij};\alpha_{kh}) $}\\
		\midrule
		N($\alpha_{kh},\sigma^{2}$)\tnote{*}& $\mathbf{X}$ &$\dfrac{1}{2\sigma^{2}} \bm{\alpha}$&$\dfrac{1}{2\sigma^{2}} \bm{\alpha}^{2}$&$\mathbf{E}_{mn}$&$\frac{1}{\sqrt{2 \pi \sigma^{2}}}  e^{-\frac{1}{2 \sigma^{2}}\left(x_{i j}-\alpha_{kh}\right)^{2}}$\\
        Ber($\alpha_{kh}$)	& $\mathbf{X}$ &$ \log_{\frac{\bm{\alpha}}{\mathbf{E}_{gs} - \bm{\alpha}}}$&$-\log_{\mathbf{E}_{gs} - \bm{\alpha}}$&$\mathbf{E}_{mn}$&$\left(\alpha_{kh}\right)^{x_{i j}}\left(1-\alpha_{kh}\right)^{1-x_{i j}}$\\
         Poisson($\beta_{ij}\alpha_{kh}$) &$\mathbf{X}$&$\log_{\bm{\alpha}}$&$\bm{\alpha}$&$\mathbf{X}\mathbf{E}_{mn}^{\top}\mathbf{X}$&$\frac{ \left(\beta_{ij}\alpha_{kh}\right)^{x_{i j}}}{x_{ij}!}e^{\left(-\beta_{ij} \alpha_{kh}\right)}$\\
		\bottomrule
	\end{tabular*}
	\end{table}
\end{center}
