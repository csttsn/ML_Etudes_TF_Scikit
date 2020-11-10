# Machine Learning Etudes
This repository contains my machine learning etudes in Scikit-Learn and TensorFlow 2.x. They are mainly based on the exercises from the book "[Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)" by Aurélien Géron.

## Scikit-Learn Etudes
1. **Support Vector Machines:** Training SVM classifiers and a regressor
2. **Decision Trees:** Training a Decision Tree and implementing a Random Forest classifier
3. **Ensemble Learning:** Implementing simple ensemble learning and a blender (both based on Random Forest classifiers)
4. **Dimensionality Reduction:** Applying PCA, t-SNE, testing openTSNE
5. **Clustering:** Clustering Olivetti faces dataset, using clustering with classifiers, Gaussian mixture models, anomaly detection

## TensorFlow 2.x Etudes
1. **Introduction to ANN:** Training a simple ANN, using callbacks
2. **Deep Neural Networks:** Training deep neural networks, transfer learning, pre-training on an auxiliary task
3. **Custom Models and Training:** Implementing a custom layer that performs layer normalization, custom training loops
4. **Loading and Preprocessing Data:** Using TFRecord (with protobuf); using tf.data.Dataset with several variants of the IMDB reviews classifier (with a TextVectorization layer)
5. **Deep Computer Vision with CNNs:** Building a simple CNN, using transfer learning for image classification
6. **Processing Sequences Using RNNs and CNNs:** Training a SketchRNN classifier, generating JS Bach-like music
7. **Natural Language Processing with RNNs:** Reber grammar checker, encoder-decoder for converting various date formats
8. **Representation and Generative Learning Using Autoencoders and GANs:** Creating an image classifier based on the denoising autoencoder, training a variational autoencoder, GAN, and conditional GAN
9. **Reinforcement Learning:** Solving Lunar Lander using policy gradient, solving Space Invaders using TF-Agents

## How to Run
All notebooks can be run in Google Colab. To open the repository in Colab, click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csttsn/ML_Etudes_TF_Scikit/blob/main/).