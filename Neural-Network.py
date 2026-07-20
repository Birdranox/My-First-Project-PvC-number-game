import numpy as np
import pandas as pd
import pickle

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def create_batches(X, y, batch_size):
    # Create batches of data for training to improve efficiency
    batches = []
    n = len(X)
    for start_index in range(0, n, batch_size):
        end_index = min(start_index + batch_size, n)
        batch_X = X[start_index:end_index]
        batch_y = y[start_index:end_index]
        batches.append((batch_X, batch_y))

    return batches

class NeuralNetwork:

    def __init__(self, no_inputs, no_hidden, no_outputs):
        self.no_inputs = no_inputs
        self.no_hidden = no_hidden
        self.no_outputs = no_outputs

        # Initialize weights and biases
        self.W1 = np.random.randn(self.no_inputs, self.no_hidden) * 0.01
        self.b1 = np.zeros((1, self.no_hidden))
        self.W2 = np.random.randn(self.no_hidden, self.no_outputs) * 0.01
        self.b2 = np.zeros((1, self.no_outputs))
    
    def feedforward(self, X):
        # Forward pass through the network
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2
    
    def backpropagation(self, X, y):
        n = X.shape[0]
        A2_error = self.A2.copy()
        A2_error[range(n), y] -= 1
        self.W2_gradient = np.dot(self.A1.T, A2_error) / n
        self.b2_gradient = np.sum(A2_error, axis=0, keepdims=True) / n
        A1_error = np.dot(A2_error, self.W2.T)
        A1_error[self.A1 <= 0] = 0  
        self.W1_gradient = np.dot(X.T, A1_error) / n
        self.b1_gradient = np.sum(A1_error, axis=0, keepdims=True) / n
    
    def update_parameters(self, learning_rate):
        # Update weights and biases using gradients
        self.W1 -= learning_rate * self.W1_gradient
        self.b1 -= learning_rate * self.b1_gradient
        self.W2 -= learning_rate * self.W2_gradient
        self.b2 -= learning_rate * self.b2_gradient
    
    def train(self, X, y, learning_rate, batch_size, epochs):
        for _ in range(epochs):
            batches = create_batches(X, y, batch_size)
            for batch_X, batch_y in batches:
                self.feedforward(batch_X)
                self.backpropagation(batch_X, batch_y)
                self.update_parameters(learning_rate)
            print(f"Epoch {_ + 1}/{epochs} completed.")

    def save_model(self, filename):
        # Save the model parameters to a file
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_model(filename):
        # Load the model parameters from a file
        with open(filename, 'rb') as f:
            return pickle.load(f)

mnist = pd.read_csv('mnist.csv')
# Labels
y = mnist.iloc[:, 0].values

# Images
X = mnist.iloc[:, 1:].values.astype(np.float32) / 255.0

# Split into train/test
train_size = int(0.8 * len(X))

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

nn = NeuralNetwork(X_train.shape[1], 128, 10)
nn.train(X_train, y_train, learning_rate=0.01, batch_size=32, epochs=100)
nn.feedforward(X_test)
y_pred = np.argmax(nn.A2, axis=1)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

nn.save_model('NeuralNetwork.dat')