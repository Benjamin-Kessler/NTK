"""An example comparing training a neural network with its infinite-width
counterpart.

In this example we train a neural network and its infinite-width kernels.
The network is a simplified version of LeNet5 with average pooling instead
of max pooling. We use momentum and minibatching on the full MNIST dataset.
Data is loaded using tensorflow datasets.
"""

import time
from absl import flags, app
import jax.numpy as np
import neural_tangents as nt
from jax import random
import datasets
import util

flags.DEFINE_integer('batch_size_kernel', 2,
                     'Batch size for kernel construction, 0 for no batching.')
flags.DEFINE_integer('network_width', 1,
                     'Factor by which the network width is multiplied.')
flags.DEFINE_integer('batch_size_output', 80,
                     'Batch size for computing final label predictions. Must be divisible by batch_size_kernel * '
                     'number_of_devices')
flags.DEFINE_integer('max_pixel', 5,
                     'Maximal number of pixels that an image is translated with.')

FLAGS = flags.FLAGS


def main(unused_argv):
    # Load and normalize data
    print('Loading data...')
    x_train, y_train, x_test, y_test = datasets.get_dataset('mnist', n_train=60000,
                                                            permute_train=True)

    # Reformat MNIST data to 28x28x1 pictures
    x_train = np.asarray(x_train.reshape(-1, 28, 28, 1))
    x_test = np.asarray(x_test.reshape(-1, 28, 28, 1))
    print(f'Data loaded and reshaped with n_train = {x_train.shape[0]} (batch size {FLAGS.batch_size_kernel}) and '
          f'n_test = {x_test.shape[0]}.')

    # # Add random translation to images
    # x_train = util.add_translation(x_train, FLAGS.max_pixel)
    # x_test = util.add_translation(x_test, FLAGS.max_pixel)
    # print(f'Random translations by up to {FLAGS.max_pixel} pixels added')

    # # Add random translations with padding
    # x_train = util.add_padded_translation(x_train, 10)
    # x_test = util.add_padded_translation(x_test, 10)
    # print(f'Random translations with additional padding up to 10 pixels added')


    # Build the LeNet network
    init_fn, f, kernel_fn = util.build_le_net(FLAGS.network_width)
    print('Network build complete')

    # Construct the kernel function
    # Use 'store_on_device = False' for larger kernels
    kernel_fn = nt.batch(kernel_fn, device_count=-1, batch_size=FLAGS.batch_size_kernel, store_on_device=False)

    # Set start time
    start_inf = time.time()

    # Bayesian and infinite-time gradient descent inference with infinite network
    print('Starting bayesian and infinite-time gradient descent inference with infinite network')
    predict_fn = nt.predict.gradient_descent_mse_ensemble(
        kernel_fn=kernel_fn,
        x_train=x_train,
        y_train=y_train,
        diag_reg=1e-6
    )

    duration_kernel = time.time() - start_inf
    print(f'Kernel constructed in {duration_kernel} seconds.')

    # fx_test_nngp_ub, fx_test_ntk_ub = predict_fn(x_test=x_test, get=('nngp', 'ntk'))

    fx_test_nngp, fx_test_ntk = [] * x_test.shape[0], [] * x_test.shape[0]
    print('Output vector allocated.')
    print(f'Available GPU memory: {util.get_gpu_memory()} MiB')

    # Compute predictions in batches
    for i in range(x_test.shape[0] // FLAGS.batch_size_output):
        time_batch = time.time()
        start, end = i * FLAGS.batch_size_output, (i+1) * FLAGS.batch_size_output
        x = x_test[start:end]
        tmp_nngp, tmp_ntk = predict_fn(x_test=x, get=('nngp', 'ntk'))
        # tmp_ntk = predict_fn(x_test=x, get='ntk')
        duration_batch = time.time() - time_batch
        print(f'Batch {i+1} predicted in {duration_batch} seconds.')
        print(f'Available GPU memory: {util.get_gpu_memory()} MiB')
        fx_test_nngp[start:end] = tmp_nngp
        fx_test_ntk[start:end] = tmp_ntk

    fx_test_nngp = np.array(fx_test_nngp)
    fx_test_ntk = np.array(fx_test_ntk)

    # fx_test_nngp.block_until_ready()
    # fx_test_ntk.block_until_ready()

    duration_inf = time.time() - start_inf

    print(f'Inference done in {duration_inf} seconds.')

    # Print out accuracy and loss for infinite network predictions.
    loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
    util.print_summary('NNGP test', y_test, fx_test_nngp, None, loss)
    util.print_summary('NTK test', y_test, fx_test_ntk, None, loss)


if __name__ == '__main__':
    app.run(main)
