# 🐱 vs 🐶 Binary Image Classifier: A CNN Approach

## Project Overview
This repository hosts a complete deep learning pipeline designed to distinguish between images of cats and dogs. Built using **TensorFlow** and **Keras**, the project solves a classic binary classification problem in computer vision. The objective is to take raw pixel data and output a probability score indicating the class (Cat or Dog). This project demonstrates foundational concepts of Convolutional Neural Networks (CNNs), including data preprocessing, architecture design, and model evaluation.

## 🧠 Technical Architecture
The model is constructed as a sequential CNN with specific design choices:

* **Convolutional Blocks:** Three stacked blocks containing `Conv2D` layers to extract spatial features (edges, shapes) and `MaxPooling2D` layers to downsample feature maps, reducing computational complexity.
* **Feature Flattening:** A `Flatten` layer converts 2D feature maps into a 1D vector.
* **Classification Head:** A Dense layer with 512 neurons (ReLU activation) followed by a single output neuron with a **Sigmoid activation function**, ideal for binary decisions.

## ⚙️ Data Pipeline & Augmentation
To prevent overfitting on limited data, the pipeline employs robust preprocessing:

1.  **Normalization:** Pixel values are rescaled from 0-255 to a normalized range of [0, 1] for smoother gradient descent convergence.
2.  **Data Augmentation:** `ImageDataGenerator` applies real-time random transformations (rotation, zoom, shifts, flips). This forces the model to learn robust, invariant features rather than memorizing specific pixels.

## 📊 Performance
The model is trained using the **Adam optimizer** and **Binary Crossentropy loss**. It typically achieves ~70% accuracy on the validation set, with augmentation successfully mitigating overfitting.

## 🛠️ Requirements
* Python 3.x
* TensorFlow 2.x
* NumPy
* Matplotlib
