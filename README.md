Neural Network for Iris Dataset Classification

This repository contains a Python implementation of a simple neural network for classifying the Iris dataset. The code utilizes the scikit-learn library to load the Iris dataset, split it into training and testing sets, and perform one-hot encoding of the target variable.

The neural network architecture consists of an input layer, a hidden layer with sigmoid activation, and an output layer with sigmoid activation. The cross-entropy loss function is used for training the network, and gradient descent is applied for optimization.

The code includes a separate function to predict the accuracy of the trained neural network on the test set. Additionally, a loss curve is plotted to visualize the training progress.

Key Features:
- Neural network implementation using numpy
- Integration with scikit-learn for dataset loading and preprocessing
- Forward and backward propagation for training the network
- Prediction function to calculate accuracy on the test set
- Loss curve visualization

Dataset:
The Iris dataset is a popular multiclass classification dataset that consists of 150 samples, each with four features: sepal length, sepal width, petal length, and petal width. The samples are labeled with three different iris species: Setosa, Versicolor, and Virginica.

Usage:
1. Install the required dependencies (numpy, matplotlib, scikit-learn).
2. Clone the repository and navigate to the project directory.
3. Run the 'iris_neural_network.py' script.
4. The script will train the neural network, output the test accuracy, and display the loss curve.
