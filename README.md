# NTD-Algorithms
This project concerns the development of a full framework to derive efficient algorithms to compute a Nonnegative Tucker Decomposition (NTD) of a given
input nonnegative Tensor ![\Large \mathcal{X}\in](https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{X}\in) 
![\small \mathbb{R}_+^{I_1\times\ldots\timesI}](https://latex.codecogs.com/svg.latex?\small&space;\mathbb{R}_+^{I_1\times\ldots\timesI})
In the particular case of a three-way tensor of dimension I_1 \times I_2 \times I_3, computing a NTD corresponds to look for three nonnegative factors W \in \mathbb{R}_+^{I_1 \times J},  
H \in \mathbb{R}_+^{I_2 \times K}, Q \in \mathbb{R}_+^{I_3 \times L} and a core tensor \mathscr{G} with \max{J,K,L} << \min{I_1,I_2,I_3} such that  
\mathscr{X} \approx \mathscr{G} \times_1 W \times_2 H \times_3 Q
where \times_i denotes the i-mode product.

We will focus on the integration of:
  - various metrics for the cost/objective function used to assess the quality of the NTD (data fitting term)
  - constraints such as normalization
  - penalty functions in the objective to promote specific solutions such as sparsity
  - acceleration schemes
  
![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}) 
