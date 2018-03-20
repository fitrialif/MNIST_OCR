import numpy as np

class Network(object):

    def __init__(self, layer_dims):

        self.num_layers = len(layer_dims)
        self.layer_dims = layer_dims
        self.parameters = {}
        for l in range(1, self.num_layers):
            self.parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * \
                                                                                    np.sqrt(1/self.layer_dims[l-1])
            self.parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))


    def random_mini_batches(self, X, Y, mini_batch_size):

        m = X.shape[1]  # number of training examples in minibatch
        mini_batches = []

        # Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]

        # Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = int(m / mini_batch_size)  # number of mini batches of size mini_batch_size in partitionning
        k=0
        for k in range(0, num_complete_minibatches):

            mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handle end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:

            mini_batch_X = shuffled_X[:, (k + 1) * mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, (k + 1) * mini_batch_size:]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def forward_propagation_with_dropout(self, minibatch_X, keep_prob=1.0):
        caches = {'A0': minibatch_X}
        for l in range(1, self.num_layers):

            if l < self.num_layers - 1:
                caches['Z' + str(l)] = np.dot(self.parameters['W' + str(l)], caches['A' + str(l - 1)]) + self.parameters['b' + str(l)]
                caches['A' + str(l)] = sigmoid(caches['Z' + str(l)])

                caches['D' + str(l)] = np.random.rand(caches['A' + str(l)].shape[0], caches['A' + str(l)].shape[1])
                caches['D' + str(l)] = caches['D' + str(l)] < keep_prob
                caches['A' + str(l)] = np.multiply(caches['A' + str(l)], caches['D' + str(l)])
                caches['A' + str(l)] = caches['A' + str(l)] / keep_prob
            else:
                caches['Z' + str(l)] = np.dot(self.parameters['W' + str(l)], caches['A' + str(l - 1)]) + self.parameters['b' + str(l)]
                caches['A' + str(l)] = sigmoid(caches['Z' + str(l)])

        return caches['A' + str(self.num_layers - 1)], caches

    def compute_cost(self, y_hat, y):
        m = y.shape[1]
        cost = (-1. / m) * np.sum(np.sum(np.multiply(y, np.log(y_hat)) + np.multiply((1 - y), np.log(1 - y_hat))))  #///nan to num
        np.squeeze(cost)
        return cost

    def backward_propagation_with_dropout(self, minibatch_Y, y_hat, caches, keep_prob=1.0):
        grads = {}
        m = minibatch_Y.shape[1]
        dAL = y_hat - minibatch_Y

        for l in range(self.num_layers - 1, 0, -1):
            dZ = dAL * sigmoid_prime(caches['Z' + str(l)])
            grads['dW' + str(l)] = (1.0 / m) * np.dot(dZ, caches['A' + str(l - 1)].T)
            grads['db' + str(l)] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
            dAL = np.dot(self.parameters['W' + str(l)].T, dZ)
            if l > 1:
                dAL = dAL * caches['D' + str(l - 1)]
                dAL = dAL / keep_prob
        return grads

    def update_parameters_with_gd(self, grads, learning_rate):

        for l in range(1, self.num_layers):
            self.parameters['W' + str(l)] = self.parameters['W' + str(l)] - learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] = self.parameters['b' + str(l)] - learning_rate * grads['db' + str(l)]


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def sigmoid_prime(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))
