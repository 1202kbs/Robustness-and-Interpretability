# Robustness and Interpretability

Experiment codes for *Bridging Adversarial Robustness and Gradient Interpretability*.

## Results

Here we show the results from our paper along with the link to the jupyter notebook that can be used to reproduce the experiment.

### Section 2.1

Notebooks [1.1](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/1.1%20MNIST%20VAE-GAN%20Training.ipynb), [2.1](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/2.1%20FMNIST%20VAE-GAN%20Training.ipynb) and [3.1](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/3.1%20CIFAR%20VAE-GAN%20Training.ipynb) was used to train the VAE-GAN.

Notebooks [1.2](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/1.2%20MNIST%20Training.ipynb), [2.2](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/2.2%20FMNIST%20Training.ipynb) and [3.2](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/3.2%20CIFAR-10%20Training.ipynb) was used to adversarially train the networks.

Notebooks [1.3](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/1.3%20MNIST%20Gradient%20Analysis.ipynb), [2.3](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/2.3%20FMNIST%20Gradient%20Analysis.ipynb) and [3.3](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/3.3%20CIFAR-10%20Gradient%20Analysis.ipynb) was used to calculate the distance of adversarial examples to their projections onto the image manifold and to visualize the loss gradients and the adversarial examples.

* MNIST

![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/gradient_mnist_xent.jpg)

* Fashion MNIST

![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/gradient_fmnist_xent.jpg)

* CIFAR-10

![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/gradient_cifar10_xent.jpg)

### Section 2.2

Notebook [4.1](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/4.1%20Mixture%20Gaussian%20Training.ipynb) was used to test our conjecture with a 2-dimensional toy dataset.

![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/boundary_experiment.jpg)

### Section 3

Notebook [3.4](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/3.4%20CIFAR-10%20ROAR%20KAR.ipynb) was used to run ROAR and KAR for neural networks trained under various adversarial attack settings. Notebook [3.5](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/3.5%20CIFAR-10%20Result%20Analysis.ipynb) was used to analyze the relation between the strength of adversary used during training and the interpretability of gradient. It was also used to identify the trade-off between accuracy and gradient interpretability under the adversarial training framework.

ROAR XEnt                                                                                                |  KAR XEnt
:-------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:
![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/roar_xent.jpg)  |  ![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/kar_xent.jpg)