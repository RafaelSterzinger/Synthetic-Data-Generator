# GAN for generating synthetic data

## Overview

For this project we decided to take on the experimental
problem of generating unstructured synthetic datasets using a Generative Adversarial Network (GAN)
as well as evaluating and comparing the classification accuracy when training a Convolutional Neural
Network (CNN) with the generated data.
We have trained three CNNs to classify on the following kinds of data:

*  original training data
*  augmented training data
*  synthetically generated data

Afterwards we compared their classification performance in regards to the original validation data.
We have expanded on the exercise’s assignment, by including data augmentation, created by flipping,
zooming, rotating and changing the brightness level of the original training data. Comparing synthetic
generated data to data augmentation is necessary, since both approaches seem to tackle the similar
problem of a too small training set. We have constructed our analysis on the following two datasets:

*   MNIST (Numbers)
*   FIDS30 (Fruits)


These datasets were chosen because they highly differ in sample size, complexity, and amount of colour
channels (greyscale and red, blue, green). The evaluation on both datasets showed that training the
CNNs on synthetic data could lead to a higher validation accuracy faster, but also over-fit earlier.
The CNNs trained on the original training data however, still achieve the highest accuracy overall.
Data augmentation performed worse than the other approaches for training accuracy while synthetic
generation proved best. This could be explained by the many possibilities data augmentation offers,
which result in endless amounts of different images and thus could be hard for a model to learn. Con-
cerning synthetically generated data, this could be the case, because the GANs are able to transcode
only the most prominent features which the respective CNNs are able to pick up on.

## Training

For our image generation we have trained two deep networks called Generator and Discriminator which
compete against and cooperate with each other, helping each other learn. The Generator creates fake
images based on random noise, then the Discriminator evaluates on these images and gives feedback
on whether or not it recognised that the images were real. The Generator uses this feedback to create
better fake images. In a separate step, the Discriminator is trained on real and generated images, in order to teach it to distinguish between real and fake. This process continues until the Generator is able to generate images, which fool the Discriminator about 50 % of the time, which is as good as guessing.
To summarise GANs consist of two major components:

1. Generator: Creates images using the opposite of convolution (transposed convolution). Re-
ceives a random vector with length 100 as an input.

2. Discriminator: Deep CNN which predicts a probability in the interval [0, 1] of how real the
input looks to it. Instead of using max-pooling like in a typical CNN, we use strided convolution
for down-sampling.

The training performance can be observed on the following example, were we train our GAN to generate the number 0.

![Training process of the number zero](https://github.com/RafaelSterzinger/ML-Exercise-3/blob/master/plots/mnist/0_loss.png)

The resulting images throughout the training are visible in the following two GIFs.

Training the model to generate a banana.

![Training the model to generate a banana.](https://github.com/RafaelSterzinger/ML-Exercise-3/blob/master/training_fruits.gif)

Training the model to generate the number nine.

![Training the model to generate the number nine.](https://github.com/RafaelSterzinger/ML-Exercise-3/blob/master/training_mnist.gif)
