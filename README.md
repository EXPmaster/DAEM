### Quantum Error Mitigation

Error mitigation using machine learning algorithms.


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