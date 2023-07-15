import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Backpropagation
        self.dz2 = self.a2 - y
        self.dW2 = (1 / m) * np.dot(self.a1.T, self.dz2)
        self.db2 = (1 / m) * np.sum(self.dz2, axis=0)
        self.dz1 = np.dot(self.dz2, self.W2.T) * self.a1 * (1 - self.a1)
        self.dW1 = (1 / m) * np.dot(X.T, self.dz1)
        self.db1 = (1 / m) * np.sum(self.dz1, axis=0)

        # Gradient descent
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode the target variable
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the neural network
input_size = X_train.shape[1]
hidden_size = 4
output_size = y_train.shape[1]
learning_rate = 0.1
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Training loop
loss_history = []
num_iterations = 200

for i in range(num_iterations):
    nn.forward(X_train)
    nn.backward(X_train, y_train, learning_rate)
    loss = cross_entropy_loss(y_train, nn.forward(X_train))
    loss_history.append(loss)

# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = y_true.shape[0]
    accuracy = correct_predictions / total_predictions
    return accuracy

# Predict on the test set
y_pred_test = nn.predict(X_test)
accuracy = calculate_accuracy(np.argmax(y_test, axis=1), y_pred_test)
print("Test Accuracy:", accuracy)

# Plot the loss curve
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()
