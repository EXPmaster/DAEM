# LBEM_for_VQE
Entry for QHack Hackathon 2022

### Project Description: 

Variational Quantum Eigensolvers (VQE) for calculating ground state energies of molecules are one of the major applications of noisy intermediate scale quantum (NISQ) computers. However for VQE to be viable on NISQ computers, powerful error mitigation protocols are needed due to the high level of noise.

In this project, we investigate applications of a learning based quantum error mitigation (LBEM) method [[1]](https://doi.org/10.1103/PRXQuantum.2.040330) on VQE for molecular ground state energy calculation. LBEM models an error free result with a quasi probabilistic mixture of noisy results. This distribution is learned via an _ab initio_ process, without prior knowledge on the hardware error model. Clifford circuits are used for the training, so classical simulation is efficient, and the mitigation takes account of both spatial and temporal correlations.

We have implemented LBEM for running H_2 and LiH ansatze on noisy hardware and simulators. Also, we have analyzed the performance of LBEM when truncating the training sets by varying the input size. Result shows successful error mitigation on both noise models and hardwares.

[[1] Strikis, Armands, et al. "Learning-based quantum error mitigation." PRX Quantum 2.4 (2021): 040330.](https://doi.org/10.1103/PRXQuantum.2.040330)

### Presentation: 
[Presentation Slides](https://docs.google.com/presentation/d/1APkuSyKE1_9k7hti1yeiNjLVo_kmxJ_YFD8aDVKGvWw/edit?usp=sharing)

### Source Codes Included:
expval_calc_q_optim.py : 
  - caculates expectation values
  - finds optimum quasiprobability distribution

generate_training_set.py :
  - generates training and error-mitigated circuits for learning

truncation_accumulated.py :
  - analyzes performance of LBEM on truncated train set

util.py :
  - generates Pauli Hamiltonian of molecule
  - runs VQE and plots the resulting PES curve
![lbem_diagram](https://user-images.githubusercontent.com/54147950/155768837-adc182b0-1685-4da3-b454-5f395498554d.png)

