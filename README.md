# CIFAR-10 CNN Model

This project demonstrates a Convolutional Neural Network (CNN) model built to classify images from the CIFAR-10 dataset. The model uses three convolutional layers followed by a fully connected layer to predict the class of an image.

## Project Overview

- **Dataset**: CIFAR-10 (Contains 60,000 32x32 color images in 10 classes, with 6,000 images per class)
- **Model**: A CNN built using TensorFlow/Keras with 3 convolutional layers and 2 max-pooling layers.
- **Training**: The model is trained for 10 epochs, and performance is evaluated on the test set.
- **Test Accuracy**: ~71.83%

## How to Run

1. Clone this repository.
2. Install dependencies: pip install -r requirements.txt
3. Run the training script: python train_model.py
   
## Results

- The model achieves a test accuracy of around **71.83%**.
- Training history and confusion matrix are plotted and saved in the project directory
- Model is downloaded and saved in the project directory

## Requirements

- TensorFlow >= 2.0
- matplotlib
- seaborn
- numpy
- scikit-learn

