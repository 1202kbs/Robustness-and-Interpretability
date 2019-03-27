# Robustness and Interpretability

Experiment codes for *Bridging Adversarial Robustness and Gradient Interpretability*.

We summarize the results from our paper along with the link to the jupyter notebook that can be used to reproduce the experiment.

## Section 2.1. Adversarial Training Confines Gradient to Image Manifold

**Section Summary**

To identify why adversarial training increases the perceptual quality of loss gradients, we hypothesized adversarial examples from adversarially trained networks lie closer to the image manifold. Note that the loss gradient is the difference between the original image and the adversarial image. Hence if the adversarial image lies closer to the image manifold, the loss gradient should align better with human perception. To see this, we used VAE-GANs to indirectly calculate the distance between an image (both natural and adversarial) and its projection onto the approximated image manifold.

**Experiment Procedure**

1. Notebooks [1.1](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/1.1%20MNIST%20VAE-GAN%20Training.ipynb), [2.1](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/2.1%20FMNIST%20VAE-GAN%20Training.ipynb) and [3.1](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/3.1%20CIFAR%20VAE-GAN%20Training.ipynb) were used to train the VAE-GAN.

2. Notebooks [1.2](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/1.2%20MNIST%20Training.ipynb), [2.2](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/2.2%20FMNIST%20Training.ipynb) and [3.2](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/3.2%20CIFAR-10%20Training.ipynb) were used to adversarially train the networks.

3. Notebooks [1.3](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/1.3%20MNIST%20Gradient%20Analysis.ipynb), [2.3](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/2.3%20FMNIST%20Gradient%20Analysis.ipynb) and [3.3](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/3.3%20CIFAR-10%20Gradient%20Analysis.ipynb) were used to calculate the distance of adversarial examples to their projections onto the image manifold and to visualize the loss gradients and the adversarial examples.

**Results**

We observed that for all datasets, adversarial examples for adversarially trained DNNs lie closer to the image manifold than those for standard DNNs. This suggests that adversarial training restricts loss gradients to the image manifold. Hence gradients from adversarially trained DNNs are more visually interpretable. Interestingly, we also observed that using stronger attacks during training reduces the distance between adversarial examples and their projections even further.

MNIST
:----------------------------------------------------------------------------------------------------------------:
![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/gradient_mnist_xent.jpg)

Fashion MNIST
:-----------------------------------------------------------------------------------------------------------------:
![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/gradient_fmnist_xent.jpg)

CIFAR-10
:-----------------------------------------------------------------------------------------------------------------:
![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/gradient_cifar10_xent.jpg)

## Section 2.2. A Conjecture Under the Boundary Tilting Perspective

**Section Summary**

In this section, we proposed a conjecture for why adversarial training conﬁnes the gradient to the manifold. Basically, our conjecture is based on the boundary tilting perspective on the phenomenon of adversarial examples. Adversarial examples exist because the decision boundary of a standard DNN is “tilted” along directions of low variance in the data (standard decision boundary). Under certain situations, the decision boundary will lie close to the data such that a small  erturbation directed toward the boundary will cause the data to cross the boundary. Moreover, since the decision boundary is tilted, the perturbed data will leave the image manifold.

We conjectured adversarial training removes such tilting along directions of low variance in the data (robust decision boundary). Intuitively, this makes sense because a network is robust when only large-*epsilon* attacks are able to cause a nontrivial drop in accuracy, and this  appens when data points of different classes are mirror images of each other with respect to the decision boundary. Since the loss gradient is generally perpendicular to the decision boundary, it will be conﬁned to the image manifold and thus adversarial examples stay within the image manifold.

As a sanity check, we tested our conjecture with a 2-dimensional toy dataset. Speciﬁcally, we trained three two-layer ReLU network to classify points from two distinct bivariate Gaussian distributions. The ﬁrst network is trained on original data and the latter two networks are trained against weak and strong adversaries. We then compared the resulting decision boundaries and the distribution of adversarial examples.

**Experiment Procedure**

Notebook [4.1](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/4.1%20Mixture%20Gaussian%20Training.ipynb) was used to test our conjecture with a 2-dimensional toy dataset.

**Results**

Illustration of our conjecture                                                                                 | Results on a 2-dimensional toy dataset
:-------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------:
![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/boundary_theory.jpg)  |  ![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/boundary_experiment.jpg)

## Section 3.2. Effect of Adversarial Training on Loss Gradient Interpretability

**Section Summary**

**Experiment Procedure**

1. Notebook [3.2](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/3.2%20CIFAR-10%20Training.ipynb) was used to train the network under various adversarial attack settings.

2. Notebook [3.4](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/3.4%20CIFAR-10%20ROAR%20KAR.ipynb) was used to run ROAR and KAR for neural networks trained under various adversarial attack settings.

3. Notebook [3.5](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/3.5%20CIFAR-10%20Result%20Analysis.ipynb) was used to analyze the relation between the strength of adversary used during training and the interpretability of gradient. We observed that adversarial training generally enhances the interpretability of attributions.

**Results**

ROAR                                                                                                     |  KAR
:-------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:
![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/roar_xent.jpg)  |  ![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/kar_xent.jpg)
![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/roar_cw.jpg)    |  ![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/kar_cw.jpg)

## Section 3.3. Accuracy and Loss Gradient Interpretability Trade-off

**Section Summary**

We were also able to discover a trade-off between test accuracy and gradient interpretability under the adversarial training framework.

**Experiment Procedure**

We reused the codes in Section 3.2.

**Results**

ROAR                                                                                                              |  KAR
:----------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------:
![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/roar_xent_tradeoff.jpg)  |  ![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/kar_xent_tradeoff.jpg)
![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/roar_cw_tradeoff.jpg)    |  ![alt tag](https://github.com/1202kbs/Robustness-and-Interpretability/blob/master/assets/kar_cw_tradeoff.jpg)