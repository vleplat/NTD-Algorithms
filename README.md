# NTD-Algorithms
This project concerns the development of a full framework to derive efficient algorithms to compute a Nonnegative Tucker Decomposition (NTD) of a given
input nonnegative Tensor ![\Large \mathcal{X}\in](https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{X}\in) 
![\large \mathbb{R}_+^{I_1\times...I_N}](https://latex.codecogs.com/svg.latex?\large&space;\mathbb{R}_+^{I_1\times...I_N}).
In the particular case of a three-way tensor of dimension ![\small I_1](https://latex.codecogs.com/svg.latex?\small&space;I_1) ![\small \times](https://latex.codecogs.com/svg.latex?\small&space;\times) ![\small I_2](https://latex.codecogs.com/svg.latex?\small&space;I_2) ![\small \times](https://latex.codecogs.com/svg.latex?\small&space;\times) ![\small I_3](https://latex.codecogs.com/svg.latex?\small&space;I_3)
computing a NTD corresponds to look for three nonnegative factors 
![\Large W\in](https://latex.codecogs.com/svg.latex?\Large&space;W\in) 
![\large \mathbb{R}_+^{I_1\times.J}](https://latex.codecogs.com/svg.latex?\large&space;\mathbb{R}_+^{I_1\times.J}),
![\Large H\in](https://latex.codecogs.com/svg.latex?\Large&space;H\in) 
![\large \mathbb{R}_+^{I_2\times.K}](https://latex.codecogs.com/svg.latex?\large&space;\mathbb{R}_+^{I_2\times.K}),
![\Large Q\in](https://latex.codecogs.com/svg.latex?\Large&space;Q\in) 
![\large \mathbb{R}_+^{I_3\times.L}](https://latex.codecogs.com/svg.latex?\large&space;\mathbb{R}_+^{I_3\times.L}),
and a nonnegative core tensor 
![\Large \mathcal{G}](https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{G}) of dimension ![\small J](https://latex.codecogs.com/svg.latex?\small&space;J) ![\small \times](https://latex.codecogs.com/svg.latex?\small&space;\times) ![\small K](https://latex.codecogs.com/svg.latex?\small&space;K) ![\small \times](https://latex.codecogs.com/svg.latex?\small&space;\times) ![\small L](https://latex.codecogs.com/svg.latex?\small&space;L)
with \max{J,K,L} << \min{I_1,I_2,I_3} such that  
\mathscr{X} \approx \mathscr{G} \times_1 W \times_2 H \times_3 Q
where \times_i denotes the i-mode product.

We will focus on the integration of:
  - various metrics for the cost/objective function used to assess the quality of the NTD (data fitting term)
  - constraints such as normalization
  - penalty functions in the objective to promote specific solutions such as sparsity
  - acceleration schemes
  
![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}) 
