"""A collection of utility functions for generating outputs."""
import jax
import jax.numpy as np
from jax import random
from jax import ops
from neural_tangents import stax
import subprocess as sp


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


def output_in_batches(x, params, f_x, number_of_batches):
    """ Compute output of function f_x in number_of_batches batches. """
    output = []
    batch_size = x.shape[0] / number_of_batches
    for i in range(number_of_batches):
        start = int(i * batch_size)
        end = int((i + 1) * batch_size)
        x_batch = x[start:end]
        output.extend(f_x(params, x_batch))
    return np.array(output)


def build_le_net(network_width):
    """ Construct the LeNet of width network_width with average pooling using neural tangent's stax."""
    return stax.serial(
        stax.Conv(out_chan=6 * network_width, filter_shape=(3, 3), strides=(1, 1), padding='VALID'),
        stax.Relu(),
        stax.AvgPool(window_shape=(2, 2), strides=(2, 2)),
        stax.Conv(out_chan=16 * network_width, filter_shape=(3, 3), strides=(1, 1), padding='VALID'),
        stax.Relu(),
        stax.AvgPool(window_shape=(2, 2), strides=(2, 2)),
        stax.Flatten(),
        stax.Dense(120 * network_width),
        stax.Relu(),
        stax.Dense(84 * network_width),
        stax.Relu(),
        stax.Dense(10)
    )


def get_gpu_memory():
    """ Print the available memory of all provided GPUs. Requires GPUs to exist. """
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    # ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return f'{memory_free_values} MiB'


def add_translation(dataset, pixel):
    """ Modify an array of images (JAX arrays) by adding random translations up to 'pixel' pixels."""
    dataset_padded = np.pad(dataset, ((0, 0), (pixel, pixel), (pixel, pixel), (0, 0)), constant_values=dataset[0, 0, 0])
    for n in range(0, dataset.shape[0]):
        key = random.PRNGKey(n)
        dataset = jax.ops.index_update(dataset, n, np.roll(dataset_padded[n], random.randint(key, [2], -pixel, pixel),
                                                           axis=(0, 1))[pixel:(28+pixel), pixel:(28+pixel)])
    return dataset


def add_padded_translation(dataset, pixel):
    """ Modify an array of images (JAX arrays) by adding padding and random translations up to 'pixel' pixels. The
    images are rescaled to their initial size. """
    dataset_padded = np.pad(dataset, ((0, 0), (pixel, pixel), (pixel, pixel), (0, 0)), constant_values=dataset[0, 0, 0])
    for n in range(0, dataset.shape[0]):
        key = random.PRNGKey(n)
        temp = np.roll(dataset_padded[n], random.randint(key, [2], -pixel, pixel), axis=(0, 1))
        dataset = jax.ops.index_update(dataset, n, jax.image.resize(temp, [28, 28, 1], "cubic"))
    return dataset
