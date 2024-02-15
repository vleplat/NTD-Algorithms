# Regularized scale-invariant low-rank approximation problems
This project concerns the development of a full framework to derive efficient algorithms to compute a regularized low-rank approximation of a given
input (nonnegative) matrix or tensor ![\Large \mathcal{X}\in](https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{X}\in) 
![\large \mathbb{R}_+^{I_1\times...I_N}](https://latex.codecogs.com/svg.latex?\large&space;\mathbb{R}_+^{I_1\times...I_N}).
In the particular case of a three-way tensor of dimension ![\small I_1](https://latex.codecogs.com/svg.latex?\small&space;I_1) ![\small \times](https://latex.codecogs.com/svg.latex?\small&space;\times) ![\small I_2](https://latex.codecogs.com/svg.latex?\small&space;I_2) ![\small \times](https://latex.codecogs.com/svg.latex?\small&space;\times) ![\small I_3](https://latex.codecogs.com/svg.latex?\small&space;I_3), we might for instance 
compute a sparse NTD, which corresponds to looking for three nonnegative factors 
![\Large W\in](https://latex.codecogs.com/svg.latex?\Large&space;W\in) 
![\large \mathbb{R}_+^{I_1\times.J}](https://latex.codecogs.com/svg.latex?\large&space;\mathbb{R}_+^{I_1\times.J}),
![\Large H\in](https://latex.codecogs.com/svg.latex?\Large&space;H\in) 
![\large \mathbb{R}_+^{I_2\times.K}](https://latex.codecogs.com/svg.latex?\large&space;\mathbb{R}_+^{I_2\times.K}),
![\Large Q\in](https://latex.codecogs.com/svg.latex?\Large&space;Q\in) 
![\large \mathbb{R}_+^{I_3\times.L}](https://latex.codecogs.com/svg.latex?\large&space;\mathbb{R}_+^{I_3\times.L}),
and a sparse nonnegative core tensor 
![\Large \mathcal{G}](https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{G}) of dimension ![\small J](https://latex.codecogs.com/svg.latex?\small&space;J) ![\small \times](https://latex.codecogs.com/svg.latex?\small&space;\times) ![\small K](https://latex.codecogs.com/svg.latex?\small&space;K) ![\small \times](https://latex.codecogs.com/svg.latex?\small&space;\times) ![\small L](https://latex.codecogs.com/svg.latex?\small&space;L)
with 
![\large \text{max}\left(J,K,L\right)\ll\text{min}\left(I_1,I_2,I_3\right)](https://latex.codecogs.com/svg.latex?\large&space;\text{max}\left(J,K,L\right)\ll\text{min}\left(I_1,I_2,I_3\right))
such that 
![\Large \mathcal{X}](https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{X}) ![\Large \approx](https://latex.codecogs.com/svg.latex?\Large&space;\approx) ![\Large \mathcal{G}](https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{G}) ![\Large \times_1](https://latex.codecogs.com/svg.latex?\Large&space;\times_1) ![\Large W](https://latex.codecogs.com/svg.latex?\Large&space;W) ![\Large \times_2](https://latex.codecogs.com/svg.latex?\Large&space;\times_2) ![\Large H](https://latex.codecogs.com/svg.latex?\Large&space;H) ![\Large \times_3](https://latex.codecogs.com/svg.latex?\Large&space;\times_3) ![\Large Q](https://latex.codecogs.com/svg.latex?\Large&space;Q)

where ![\Large \times_i](https://latex.codecogs.com/svg.latex?\Large&space;\times_i) denotes the i-mode product.

This repository contains only code tailored for KL loss, the rest of the code (l2 loss) is in a branch of tensorly. Our implementation integrates
  - various metrics for the cost/objective function used to assess the quality of the approximation (data fitting term)
  - homogeneous regularizations to promote specific solutions such as sparsity.
  - nonnegativity is optional.
  

## Installation

Installation of this package is tricky because of a dependency on a specific branch of a fork of tensorly. We will start by installing this. Visit this repository: [link](https://github.com/cohenjer/tensorly/tree/HRSI_draft), clone it and perform a local installation with `pip install -e .`.

Now we can install the rest of the dependencies. Clone the current repository, and run `pip install -r requirements.txt`. You should be good to go.


