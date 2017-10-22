import my_network
import numpy as np
import my_mnist_loader
import matplotlib.pyplot as plt


def predict(X, Y, obj):
    #print(X.shape,Y.shape)
    a, _ = obj.forward_propagation_with_dropout(X)
    result = [(np.argmax(a[:,i]),np.argmax(Y[:,i])) for i in range(10000)]
    return sum(int(x == y) for (x, y) in result)


def get_dset(data):
    X, Y = zip(*data)
    X = np.array(X)
    Y = np.array(Y)
    #print(X.shape,Y.shape)
    X = X.reshape((X.shape[0], X.shape[1])).T
    Y = Y.reshape((Y.shape[0], Y.shape[1])).T

    return (X, Y)


def model(layer_dims, learning_rate=0.3, mini_batch_size=64, beta1=0.9, beta2=0.999,
          epsilon=1e-8, num_epochs=400, keep_prob=0.80, print_cost=True):

    training_data, validation_data, test_data = my_mnist_loader.load_data_wrapper()

    X, Y = get_dset(training_data)
    c, d = get_dset(validation_data)
    costs = []
    adam_counter = 0
    net = my_network.Network(layer_dims)
    #v, s = net.initialize_adam()

    for i in range(num_epochs):

        cost = 0
        minibatches = net.random_mini_batches(X, Y, mini_batch_size)

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            a, caches = net.forward_propagation_with_dropout(minibatch_X, keep_prob)
            cost = net.compute_cost(a, minibatch_Y)
            grads = net.backward_propagation_with_dropout(minibatch_Y, a, caches, keep_prob)
            #adam_counter = adam_counter + 1
            #net.update_parameter_with_adam(grads, v, s, adam_counter, learning_rate, beta1, beta2, epsilon)
            net.update_parameters_with_gd(grads, learning_rate)

        if print_cost and i%10 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
            print("validation set : %d/10000" % predict(c, d, net))
            #print("train set : %d/10000"%predict(X[:, 0:10000], Y[:, 0:10000], net))

        if print_cost and i % 10 == 0:
            costs.append(cost)

    A, B = get_dset(test_data)
    print("test set : %d/10000" % predict(A, B, net))
    print("train set : %d/10000" % predict(X[:, 0:10000], Y[:, 0:10000], net))

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 10)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()



def param_search(a=1.0,b=0.0001):
    param_list = np.random.rand(1,10)


model([784, 60, 30, 10])
