# Algorithms for Regularized Nonnegative Scale-invariant Low-rank Approximation Models
This project concerns the development of a full framework to derive efficient algorithms to compute a regularized low-rank approximation of a given
input (nonnegative) matrix or tensor $\mathcal{X}\in 
\mathbb{R}_+^{I_1\times...I_N}$.
In the particular case of a three-way tensor of dimension $I_1\times I_2 \times I_3$, we might for instance 
compute a sparse NTD, which corresponds to looking for three nonnegative factors 
$ W\in \mathbb{R}_+^{I_1\times.J},  H\in \mathbb{R}_+^{I_2\times.K}, Q\in \mathbb{R}_+^{I_3\times.L}$
and a sparse nonnegative core tensor 
$\mathcal{G}$ of dimension $J\times K\times L$
with 
$\text{max}\left(J,K,L\right)\ll\text{min}\left(I_1,I_2,I_3\right)$
such that 
$\mathcal{X}\approx \mathcal{G} \times_1 W \times_2 H \times_3 Q$

where $\times_i$ denotes the i-mode product.

This repository contains only code tailored for KL loss, the rest of the code (l2 loss) is in a branch of tensorly. Our implementation integrates
  - various metrics for the cost/objective function used to assess the quality of the approximation (data fitting term)
  - homogeneous regularizations to promote specific solutions such as sparsity.
  - nonnegativity is optional.
  

## Installation

Installation of this package is tricky because of a dependency on a specific branch of a fork of tensorly. We will start by installing this. Visit this repository: [link](https://github.com/cohenjer/tensorly/tree/HRSI_draft), clone it and perform a local installation with `pip install -e .`.

Now we can install the rest of the dependencies. Clone the current repository, and run `pip install -r requirements.txt`. You should be good to go.


