"""An example comparing training a neural network with its linearization and
its infinite-width counterpart.

In this example we train a neural network, its infinite-width kernels and a
linear model corresponding to the first order Taylor seres of the network
about its initial parameters. The network is a simplified version of LeNet5
with average pooling instead of max pooling. We use momentum and minibatching
on the full MNIST dataset. Data is loaded using tensorflow datasets.
"""

import time
from absl import app
from absl import flags
from jax import random
from jax.api import grad
from jax.api import jit
from jax.experimental import optimizers
from jax.experimental.stax import logsoftmax
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util

flags.DEFINE_float('learning_rate', 0.1,
                   'Learning rate to use during training.')
flags.DEFINE_integer('batch_size', 100,
                     'Batch size to use during training.')
flags.DEFINE_integer('batch_size_kernel', 10,
                     'Batch size for kernel construction, 0 for no batching.')
flags.DEFINE_integer('train_epochs', 100,
                     'Number of epochs to train for.')
flags.DEFINE_integer('network_width', 1,
                     'Factor by which the network width is multiplied.')

FLAGS = flags.FLAGS


def main(unused_argv):
    # Load and normalize data
    print('Loading data...')
    x_train, y_train, x_test, y_test = datasets.get_dataset('mnist',
                                                            permute_train=True)

    # Reformat MNIST data to 28x28x1 pictures
    x_train = np.asarray(x_train.reshape(-1, 28, 28, 1))
    x_test = np.asarray(x_test.reshape(-1, 28, 28, 1))
    print('Data loaded and reshaped')

    # Build the LeNet network
    init_fn, f, kernel_fn = stax.serial(
        stax.Conv(out_chan=6 * FLAGS.network_width, filter_shape=(3, 3), strides=(1, 1), padding='VALID'),
        stax.Relu(),
        stax.AvgPool(window_shape=(2, 2), strides=(1, 1)),
        stax.Conv(out_chan=16 * FLAGS.network_width, filter_shape=(3, 3), strides=(1, 1), padding='VALID'),
        stax.Relu(),
        stax.AvgPool(window_shape=(2, 2), strides=(1, 1)),
        stax.Flatten(),
        stax.Dense(120 * FLAGS.network_width),
        stax.Relu(),
        stax.Dense(84 * FLAGS.network_width),
        stax.Relu(),
        stax.Dense(10)
    )
    print('Network build complete')

    # Construct the kernel function
    kernel_fn = nt.batch(kernel_fn, device_count=-1, batch_size=FLAGS.batch_size_kernel)
    print('Kernel constructed')

    # Compute random initial parameters
    key = random.PRNGKey(0)
    _, params = init_fn(key, (-1, 28, 28, 1))

    # Linearize the network about its initial parameters.
    f_lin = nt.linearize(f, params)

    # Create and initialize an optimizer for both f and f_lin.
    opt_init, opt_apply, get_params = optimizers.momentum(FLAGS.learning_rate, 0.9)
    opt_apply = jit(opt_apply)

    # Compute the initial states
    state = opt_init(params)
    state_lin = opt_init(params)

    # Create a cross-entropy loss function.
    loss = lambda fx, y_hat: -np.mean(logsoftmax(fx) * y_hat)

    # Specialize the loss function to compute gradients for both linearized and
    # full networks.
    grad_loss = jit(grad(lambda params, x, y: loss(f(params, x), y)))
    grad_loss_lin = jit(grad(lambda params, x, y: loss(f_lin(params, x), y)))

    # Train the network.
    print('Training...')
    print('Epoch\tTime\tLoss\tLinearized Loss')
    print('------------------------------------------')

    epoch = 1
    steps_per_epoch = x_train.shape[0] // FLAGS.batch_size

    start = time.time()
    start_epoch = time.time()

    for i, (x, y) in enumerate(datasets.minibatch(
            x_train, y_train, FLAGS.batch_size, FLAGS.train_epochs)):

        params = get_params(state)
        state = opt_apply(i, grad_loss(params, x, y), state)

        params_lin = get_params(state_lin)
        state_lin = opt_apply(i, grad_loss_lin(params_lin, x, y), state_lin)

        if (i+1) % steps_per_epoch == 0:
            time_point = time.time() - start_epoch
            print('{}\t{:.3f}\t{:.4f}\t{:.4f}'.format(
                epoch, time_point, loss(f(params, x), y), loss(f_lin(params_lin, x), y)))
            epoch += 1
            start_epoch = time.time()

    duration = time.time() - start
    print('------------------------------------------')
    print(f'Training complete in {duration} seconds.')

    # Print out summary data comparing the linear / nonlinear model.
    x, y = x_train[:10000], y_train[:10000]
    util.print_summary('train', y, f(params, x), f_lin(params_lin, x), loss)
    util.print_summary(
        'test', y_test, f(params, x_test), f_lin(params_lin, x_test), loss)


if __name__ == '__main__':
    app.run(main)
