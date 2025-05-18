import numpy as np

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)] + 1e-9)
    return np.sum(log_likelihood) / m

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def initialize_weights(layer_sizes):
    weights = {}
    biases = {}
    for i in range(len(layer_sizes) - 1):
        weights[f'W{i + 1}'] = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
        biases[f'b{i + 1}'] = np.zeros((1, layer_sizes[i + 1]))
    return weights, biases

def forward_pass(X, weights, biases):
    cache = {'A0': X}
    num_layers = len(weights)
    for i in range(1, num_layers + 1):
        cache[f'Z{i}'] = np.dot(cache[f'A{i - 1}'], weights[f'W{i}']) + biases[f'b{i}']
        if i < num_layers:
            cache[f'A{i}'] = tanh(cache[f'Z{i}'])
        else:
            cache[f'A{i}'] = softmax(cache[f'Z{i}'])
    return cache

def backward_pass(X, y, cache, weights, biases):
    grads = {}
    m = X.shape[0]
    num_layers = len(weights)
    dZ = cache[f'A{num_layers}'] - y
    grads[f'dW{num_layers}'] = np.dot(cache[f'A{num_layers - 1}'].T, dZ) / m
    grads[f'db{num_layers}'] = np.sum(dZ, axis=0, keepdims=True) / m
    for i in range(num_layers - 1, 0, -1):
        dA = np.dot(dZ, weights[f'W{i + 1}'].T)
        dZ = dA * tanh_derivative(cache[f'Z{i}'])
        grads[f'dW{i}'] = np.dot(cache[f'A{i - 1}'].T, dZ) / m
        grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m
    return grads

def train(X_train, y_train, layer_sizes, epochs=10, batch_size=64, learning_rate=0.01):
    weights, biases = initialize_weights(layer_sizes)
    y_train_onehot = one_hot_encode(y_train, layer_sizes[-1])
    num_samples = X_train.shape[0]
    history = {'loss': [], 'accuracy': []}
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_onehot[indices]
        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            cache = forward_pass(X_batch, weights, biases)
            grads = backward_pass(X_batch, y_batch, cache, weights, biases)
            for key in weights:
                weights[key] -= learning_rate * grads[f'd{key}']
                biases[f'b{key[1:]}'] -= learning_rate * grads[f'db{key[1:]}']
        cache = forward_pass(X_train, weights, biases)
        loss = cross_entropy_loss(y_train_onehot, cache[f'A{len(weights)}'])
        accuracy = np.mean(cache[f'A{len(weights)}'].argmax(axis=1) == y_train)
        history['loss'].append(loss)
        history['accuracy'].append(accuracy)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return weights, biases, history

def predict(X, weights, biases):
    cache = forward_pass(X, weights, biases)
    return cache[f'A{len(weights)}'].argmax(axis=1)