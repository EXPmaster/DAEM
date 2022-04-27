## Quantum Error Mitigation

This project aims at mitigating errors for NISQ quantum circuits caused by noise using deep learning algorithms: 1. GAN; 2. Supervised ANN with MSE loss. We assume that the circuit layout and error model is fixed, while the observable is alterable.



### Methodology

#### Circuit Layout

The following circuit is random generated and is used to train the mitigation model. Random quantum gates are sampled from set of single qubit gates - `{X, Y, Z, H, S, T, RX, RY, RZ} ` , and two-qubit gates - `{CNOT, CZ}`.

![img]()

A mitigation gate (denoted by $\mathbf{P}$) is inserted before and after each gate. Each mitigation gate is one of 16 basis operations sampled from a distribution.

<img src="https://github.com/EXPmaster/QuantumErrorMitigation/raw/master/imgs/basis_ops.png" alt="img" style="zoom:50%;" /> 

Depolarizing noise is applied after each single qubit gate, two-qubit gate and mitigation gate:
$$
\mathcal E(\rho)=(1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z),
$$
where $p=0.01$.



#### Model

The whole framework consists of two models:

* A surrogate model;

* A generator model.

The generator model is used to generate a probability distribution of 16 basis operations of each mitigation gate, while the surrogate model maps the distribution to noisy measurement result. To achieve error mitigaiton, we must generate a proper distribution such that the noisy measurement result predicted by the surrogate model is close to noise-free result.

* Surrogate model

  ![img](https://github.com/EXPmaster/QuantumErrorMitigation/raw/master/imgs/surrogate_model.png)

* Generator model

  <img src="https://github.com/EXPmaster/QuantumErrorMitigation/raw/master/imgs/generator.png" alt="img" style="zoom:50%;" />

* GAN

  * We need an additional discriminator model, which output the probability of  the input measurement result being noise-free.

  <img src="https://github.com/EXPmaster/QuantumErrorMitigation/raw/master/imgs/discriminator.png" alt="img" style="zoom:50%;" />



#### Training Strategy

1. Randomly generate some distributions and observables. Construct mitigation gates using gates sampled from the 16 basis operations according to the distributions. Then run noisy simulation. Each simulation uses newly sampled mitigation gates and measures once. Meanwhile, run ideal simulation given the original circuit layout (without mitigation gates) and the corresponding observable.
2. Train the surrogate model using the noisy simulation data, minimizing the $L_2$ distance between surrogate model output and noisy measurement result under the generated distribution.
3. Train a supervised model by minimizing the $L_2$ distance between the output of the surrogate model and the ideal simulation data. Or alternatively, introduce a discriminator which tells whether the generated data is noisy, and train a GAN.

`TODO: algorithm pseudocode` 

---

### Benchmark & Performance

#### Performance on random circuit

Validation resulst evaluated on the testset of the generated circuit, where the metric is mean absolute deviation:
$$
D(\mathbf y_{\mathrm{pred}}, \mathbf y_{\mathrm{true}}) = \frac{1}{n}\sum_{i=1}^n \left|y^{(i)}_{\mathrm{pred}} - y^{(i)}_{\mathrm{true}}\right|
$$

|   Model    | Absolute Deviation | Mitigation Ratio |
| :--------: | :----------------: | :--------------: |
|    None    |      0.179332      |        -         |
| Supervised |      0.002401      |      74.69       |
|    GAN     |      0.008902      |      20.16       |



#### Performance on IBMQ device

Validation results on IBM quantum device (ibmq_santiago).

* Benchmark circuit layout 1

  ![img](https://github.com/EXPmaster/QuantumErrorMitigation/raw/master/imgs/twoqubit_circuit.png)

* Results

  |   Model    | Absolute Deviation | Mitigation Ratio |
  | :--------: | :----------------: | :--------------: |
  |    None    |                    |        -         |
  | Supervised |                    |                  |
  |    GAN     |                    |                  |



* Benchmark circuit layout 2

  ![img](https://github.com/EXPmaster/QuantumErrorMitigation/raw/master/imgs/swaptest_circuit.png)

* Results

  |   Model    | Absolute Deviation | Mitigation Ratio |
  | :--------: | :----------------: | :--------------: |
  |    None    |                    |        -         |
  | Supervised |                    |                  |
  |    GAN     |                    |                  |

---


### Usage
To train the old model:
```
python exp_torch/train.py --model-type QuantumModelv2 --logdir runs/v2
```

GAN model is in the directory `exp_gan`. First generate some data for training surrogate model.
```
python my_envs.py
```

Next, train surrogate model with the following command:
```
python train_surrogate.py
```

Then generate data for training a mitigation model:
```
python datasets.py
```

After everything is ready, you can either train a supervised model
```
python train_mitigate.py
```

or train a GAN
```
python train_gan.py
```