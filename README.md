# <div align="center">Flexible Error Mitigation of Quantum Processes with Data Augmentation Empowered Neural Model</div>



> [** Flexible Error Mitigation of Quantum Processes with Data Augmentation Empowered
> Neural Model **]()
>
> by Manwen Liao<sup>1</sup> \*, [Yan Zhu](https://scholar.google.com/citations?user=sC4bSoEAAAAJ&hl=en)<sup>1</sup> \*, [Giulio Chiribella](https://scholar.google.com/citations?user=4ob0VU4AAAAJ&hl=en)<sup>1, 2, 3</sup>, [Yuxiang Yang](https://scholar.google.com/citations?user=jpFFDKcAAAAJ&hl=en)<sup>1 :email:</sup>
>
> <sup>1</sup> QICI Quantum Information and Computation Initiative, Department of Computer Science, The    University of Hong Kong, Pokfulam Road, Hong Kong
>
> <sup>2</sup> Department of Computer Science, Parks Road, Oxford, OX1 3QD, United Kingdom
>
> <sup>3</sup> Perimeter Institute for Theoretical Physics, Waterloo, Ontario N2L 2Y5, Canada
>
> (\*) equal contribution, (<sup>:email:</sup>) corresponding author.



## Data Augmentation Empowered Neural Model (DAEM)

### The Illustration of DAEM

<img src="https://github.com/EXPmaster/DAEM/raw/master/imgs/DAEM_framework.png" alt="daem" style="zoom:50%;" />

### Highlights of Our Framework

* Exemption from noise-free statistics reliance: Our proposed architecture eliminates the need for noise-free statistics acquired from the target quantum process. This feature allows the model to be potentially applicable in real-world experiments. Additionally, while our model still relies on noisy measurement data across different noise levels, it does not assume the knowledge of noise level values.
* Versatility on various types of quantum processes: The proposed architecture is flexible and can accept different forms of measurement statistics as inputs, making it suitable for various quantum processes, including quantum circuits, continuous-variable quantum processes, and dynamics of large-scale spin systems.
* Adaptability to diverse settings and noise models: Our data-driven model can be trained using measurement statistics without relying on rigid assumptions about noise models or requiring a specific initial state or measurement setting. Furthermore, it is capable of mitigating not only the error of observable expectations, but also the distortion of measurement probability distributions.



### Error Mitigation Results

#### Variational Quantum Eigensolvers (VQE)

<img src="https://github.com/EXPmaster/DAEM/raw/master/imgs/figure_vqe.png" alt="vqe" style="zoom:67%;" />

a. The variational ansatz for preparing the ground states of 4-qubit transverse Ising models. b. Mean Absolute Errors (MAE) between the mitigated measurement expectation values for phase damping noise model and ideal expectation values. c.Mean Absolute Errors (MAE) between the mitigated measurement expectation values for amplitude damping noise model and ideal expectation values. d. Schematic diagram of quantum circuits affected by Non-Markovian noise. e. Mean Absolute Errors (MAE) between the mitigated measurement expectation values for considered Non-Markovian noise model and ideal expectation values. 

#### Swap Test

<img src="https://github.com/EXPmaster/DAEM/raw/master/imgs/figure_swap.png" alt="swap" style="zoom:40%;" />

a. The swap test circuit for comparing two 5-qubit states. The gate within the green box is the controlled-SWAP gate. b. Mean Absolute Errors (MAE) between the mitigated fidelity values and the ground truth values.

#### Quantum Approximate Optimisation Algorithms (QAOA)

<img src="https://github.com/EXPmaster/DAEM/raw/master/imgs/figure_qaoa.png" alt="swap" style="zoom:40%;" />

a. An instance of a graph for for the Max-cut problem. b. The variational ansatz for implementing QAOA algorithm. c. Ideal, Noisy and Mitigated frequency of measurement results.

#### Spin-system dynamics

<img src="https://github.com/EXPmaster/DAEM/raw/master/imgs/figure_large.png" alt="swap" style="zoom:50%;" />

a. MAE between the mitigated measurement expectation values for phase damping noise model and ideal expectation values. b. MAE between the mitigated measurement expectation values for amplitude damping noise model and ideal expectation values.

#### Continuous-variable process

<img src="https://github.com/EXPmaster/DAEM/raw/master/imgs/figure_cv.png" alt="swap" style="zoom:70%;" />

a. Fidelity values between the noisy/mitigated state and the ideal state. b. Snapshots of the point-wise measurement results of the state at different time points.



---

### Requirement

This codebase has been developed with python 3.7, PyTorch 1.12+:

```bash
conda install pytorch==1.12.1 cudatoolkit=10.2 -c pytorch
```

See `requirements.txt` for additional requirements.

```bash
pip install -r requirements.txt
```

### Data

The data for training can be downloader from [here](https://drive.google.com/drive/folders/1XTBJeP23kFQKgbU001bCWILTV8S9lh9a?usp=share_link). Alternatively, you may refer to `src/dataset.py`, `src/error_mitigation_data`, and `src/TensornetSimulator` to generate your training data.

### Training

To train neural model for error mitigation of quantum algorithms and continuous-variable process, run:

```bash
python src/train_supervise.py --train-path /path/to/train_file --test-path /path/to/validation_file
```

To train neural model for mitigating errors in continuous-variable process, run:

```bash
python src/train_cv.py --train-path /path/to/train_file --test-path /path/to/validation_file
```

### Evaluate

The evaluation results can be downloaded from [here](https://drive.google.com/drive/folders/1XTBJeP23kFQKgbU001bCWILTV8S9lh9a?usp=share_link). You may evaluate and compare DAEM with ZNE and CDR by running `benchmark.py`.



## Citation

