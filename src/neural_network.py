import numpy as np
class NeuralNetwork:
    def __init__(self, layers, activation='relu'):
        self.activation_name = activation
        self.weights = []
        self.biases = []
        self.cache = {}
        for i in range(len(layers) - 1):
            input_dim = layers[i]
            output_dim = layers[i+1]
            limit = np.sqrt(2 / input_dim)
            self.weights.append(np.random.randn(input_dim, output_dim) * limit)
            self.biases.append(np.zeros((1, output_dim)))
    def activation(self, x):
        if self.activation_name == 'relu': return np.maximum(0, x)
        elif self.activation_name == 'sigmoid': return 1 / (1 + np.exp(-x))
        elif self.activation_name == 'tanh': return np.tanh(x)
    def activation_derivative(self, x):
        if self.activation_name == 'relu': return (x > 0).astype(float)
        elif self.activation_name == 'sigmoid': s = self.activation(x); return s * (1 - s)
        elif self.activation_name == 'tanh': return 1 - np.tanh(x) ** 2
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    def forward(self, X):
        self.cache['A0'] = X
        A = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b
            A = self.activation(Z) if i < len(self.weights) - 1 else self.softmax(Z)
            self.cache[f'Z{i+1}'] = Z
            self.cache[f'A{i+1}'] = A
        return A
    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]
        y_one_hot = np.zeros_like(output)
        y_one_hot[np.arange(m), y] = 1
        dA = output - y_one_hot
        for i in reversed(range(len(self.weights))):
            Z = self.cache[f'Z{i+1}']
            A_prev = self.cache[f'A{i}']
            if i == len(self.weights) - 1: dZ = dA
            else: dZ = dA * self.activation_derivative(Z)
            dW = A_prev.T @ dZ / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            dA = dZ @ self.weights[i].T
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
    def train(self, X, y, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if epoch % 10 == 0:
                loss = -np.mean(np.log(output[np.arange(len(y)), y] + 1e-9))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
