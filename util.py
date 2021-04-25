"""A collection of utility functions for generating outputs."""

import jax.numpy as np


def _accuracy(y, y_hat):
    """Compute the accuracy of predictions y with respect to one-hot labels."""
    return np.mean(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))


def print_summary(name, labels, net_p, lin_p, loss):
    """Print summary information of network performance.
    If lin_p is not none, information contains comparison of a network with its linearization."""
    print('\nEvaluating Network on {} data.'.format(name))
    print('---------------------------------------')
    print('Network Accuracy = {}'.format(_accuracy(net_p, labels)))
    print('Network Loss = {}'.format(loss(net_p, labels)))
    if lin_p is not None:
        print('Linearization Accuracy = {}'.format(_accuracy(lin_p, labels)))
        print('Linearization Loss = {}'.format(loss(lin_p, labels)))
        print('RMSE of predictions: {}'.format(
            np.sqrt(np.mean((net_p - lin_p) ** 2))))
    print('---------------------------------------')


def output_in_batches(x, y, params, f_x, number_of_batches):
    output = []
    batch_size = x.shape[0] / number_of_batches
    for i in range(number_of_batches):
        start = int(i*batch_size)
        end = int((i+1)*batch_size)
        x_batch, y_batch = x[start:end], y[start:end]
        output.extend(f_x(params, x_batch))
    return np.array(output)
