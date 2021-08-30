"""An example comparing training a neural network with its linearization.

In this example we train a neural network and a linear model corresponding
to the first order Taylor seres of the network about its initial parameters.
The network is a simplified version of LeNet5 with average pooling instead
of max pooling. We use momentum and minibatching on the full MNIST dataset.
Data is loaded using tensorflow datasets.
"""

import time

from absl import app
from absl import flags
from jax import random
from jax.api import grad
from jax.api import jit
from jax.experimental import optimizers
import jax.numpy as np
import neural_tangents as nt
import datasets
import util

import pdb

flags.DEFINE_float('learning_rate', 10,
                   'Learning rate to use during training.')
flags.DEFINE_integer('batch_size', 1000,
                     'Batch size to use during training.')
flags.DEFINE_integer('batch_size_kernel', 0,
                     'Batch size for kernel construction, 0 for no batching.')
flags.DEFINE_integer('train_epochs', 100000,
                     'Number of epochs to train for.')
flags.DEFINE_integer('network_width', 1,
                     'Factor by which the network width is multiplied.')
flags.DEFINE_integer('learning_decline', 30000,
                     'Number of epochs after which the learning rate is divided by 10.')
flags.DEFINE_integer('batch_count_accuracy', 1000,
                     'Number of batches when computing output over entire data set.')
flags.DEFINE_string('default_path', '/scicore/home/rothvo/kesben00/NTK/params/',
                    'Path to directory where params files should be saved.')
flags.DEFINE_integer('max_pixel', 5,
                     'Maximal number of pixels that an image is translated with.')

FLAGS = flags.FLAGS


def main(unused_argv):
    # print(f'Available GPU memory: {util.get_gpu_memory()}')
    # Load and normalize data
    print('Loading data...')
    x_train, y_train, x_test, y_test = datasets.get_dataset('mnist', n_train=60000, n_test=10000,
                                                            permute_train=True)
    # print(f'Available GPU memory: {util.get_gpu_memory()}')

    # Reformat MNIST data to 28x28x1 pictures
    x_train = np.asarray(x_train.reshape(-1, 28, 28, 1))
    x_test = np.asarray(x_test.reshape(-1, 28, 28, 1))
    print('Data loaded and reshaped')
    # print(f'Available GPU memory: {util.get_gpu_memory()}')

    # Set random seed
    key = random.PRNGKey(0)

    # # Add random translation to images
    # x_train = util.add_translation(x_train, FLAGS.max_pixel)
    # x_test = util.add_translation(x_test, FLAGS.max_pixel)
    # print(f'Random translation by up to {FLAGS.max_pixel} pixels added')

    # # Add random translations with padding
    # x_train = util.add_padded_translation(x_train, 10)
    # x_test = util.add_padded_translation(x_test, 10)
    # print(f'Random translations with additional padding up to 10 pixels added')

    # Build the LeNet network with NTK parameterization
    init_fn, f, kernel_fn = util.build_le_net(FLAGS.network_width)
    print(f'Network of width x{FLAGS.network_width} built.')

    # # Construct the kernel function
    # kernel_fn = nt.batch(kernel_fn, device_count=-1, batch_size=FLAGS.batch_size_kernel)
    # print('Kernel constructed')
    # print(f'Available GPU memory: {util.get_gpu_memory()}')

    # Compute random initial parameters
    _, params = init_fn(key, (-1, 28, 28, 1))
    params_lin = params

    print('Initial parameters constructed')
    # print(f'Available GPU memory: {util.get_gpu_memory()}')

    # # Save initial parameters
    # with open('init_params.npy', 'wb') as file:
    #     np.save(file, params)

    # Linearize the network about its initial parameters.
    # Use jit for faster GPU computation (only feasible for width < 25)
    f_lin = nt.linearize(f, params)
    if FLAGS.network_width <= 10:
        f_jit = jit(f)
        f_lin_jit = jit(f_lin)
    else:
        f_jit = f
        f_lin_jit = f_lin

    # Create a callable function for dynamic learning rates
    # Starts with learning_rate, divided by 10 after learning_decline epochs.
    dynamic_learning_rate = lambda iteration_step: FLAGS.learning_rate / 10 ** ((iteration_step //
                                                                                 (x_train.shape[0] // FLAGS.batch_size)) //
                                                                                FLAGS.learning_decline)

    # Create and initialize an optimizer for both f and f_lin.
    # Use momentum with coefficient 0.9 and jit
    opt_init, opt_apply, get_params = optimizers.momentum(dynamic_learning_rate, 0.9)
    opt_apply = jit(opt_apply)

    # Compute the initial states
    state = opt_init(params)
    state_lin = opt_init(params)

    # Define the accuracy function
    accuracy = lambda fx, y_hat: np.mean(np.argmax(fx, axis=1) == np.argmax(y_hat, axis=1))

    # Define mean square error loss function
    loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)

    # # Create a cross-entropy loss function.
    # loss = lambda fx, y_hat: -np.mean(logsoftmax(fx) * y_hat)

    # Specialize the loss function to compute gradients for both linearized and
    # full networks.
    grad_loss = jit(grad(lambda params, x, y: loss(f(params, x), y)))
    grad_loss_lin = jit(grad(lambda params, x, y: loss(f_lin(params, x), y)))

    # Train the network.
    print(f'Training with dynamic learning decline after {FLAGS.learning_decline} epochs...')
    print('Epoch\tTime\tAccuracy\tLin. Accuracy\tLoss\tLin. Loss\tAccuracy Train\tLin.Accuracy Train')
    print('----------------------------------------------------------------------------------------------------------')

    # Initialize training
    epoch = 0
    steps_per_epoch = x_train.shape[0] // FLAGS.batch_size

    # Set start time (total and 100 epochs)
    start = time.time()
    start_epoch = time.time()

    for i, (x, y) in enumerate(datasets.minibatch(
            x_train, y_train, FLAGS.batch_size, FLAGS.train_epochs)):

        # Update the parameters
        params = get_params(state)
        state = opt_apply(i, grad_loss(params, x, y), state)

        params_lin = get_params(state_lin)
        state_lin = opt_apply(i, grad_loss_lin(params_lin, x, y), state_lin)

        # Print information after each 100 epochs
        if (i + 1) % (steps_per_epoch * 100) == 0:
            time_point = time.time() - start_epoch

            # Update epoch
            epoch += 100

            # Accuracy in batches
            f_x = util.output_in_batches(x_train, params, f_jit, FLAGS.batch_count_accuracy)
            f_x_test = util.output_in_batches(x_test, params, f_jit, FLAGS.batch_count_accuracy)
            f_x_lin = util.output_in_batches(x_train, params_lin, f_lin_jit, FLAGS.batch_count_accuracy)
            f_x_lin_test = util.output_in_batches(x_test, params_lin, f_lin_jit, FLAGS.batch_count_accuracy)
            # time_point = time.time() - start_epoch

            # Print information about past 100 epochs
            print('{}\t{:.3f}\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t{:.4f}\t\t{:.4f}\t\t{:.4f}'.format(
                epoch, time_point, accuracy(f_x, y_train) * 100, accuracy(f_x_lin, y_train) * 100,
                loss(f_x, y_train), loss(f_x_lin, y_train),
                                   accuracy(f_x_test, y_test) * 100, accuracy(f_x_lin_test, y_test) * 100))

            # # Save params if epoch is multiple of learning decline or multiple of fixed value
            # if epoch % FLAGS.learning_decline == 0:
            #     filename = FLAGS.default_path + f'LinLeNetx{FLAGS.network_width}_pmod_{epoch}_{FLAGS.learning_decline}.npy'
            #     with open(filename, 'wb') as file:
            #         np.save(file, params)
            #     filename_lin = FLAGS.default_path + f'LinLeNetx{FLAGS.network_width}_pmod_{epoch}_{FLAGS.learning_decline}_lin.npy'
            #     with open(filename_lin, 'wb') as file_lin:
            #         np.save(file_lin, params_lin)

            # Reset timer
            start_epoch = time.time()

    duration = time.time() - start
    print('----------------------------------------------------------------------------------------------------------')
    print(f'Training complete in {duration} seconds.')

    # # Save final params in file
    # filename_final = FLAGS.default_path + f'LinLeNetx{FLAGS.network_width}_final_pmod_{FLAGS.train_epochs}_{FLAGS.learning_decline}.npy '
    # with open(filename_final, 'wb') as final:
    #     np.save(final, params)
    # filename_final_lin = FLAGS.default_path + f'LinLeNetx{FLAGS.network_width}_final_pmod_{FLAGS.train_epochs}_{FLAGS.learning_decline}_lin.npy'
    # with open(filename_final_lin, 'wb') as final_lin:
    #     np.save(final_lin, params_lin)

    # Compute output in batches
    f_x = util.output_in_batches(x_train, params, f_jit, FLAGS.batch_count_accuracy)
    f_x_lin = util.output_in_batches(x_train, params_lin, f_lin_jit, FLAGS.batch_count_accuracy)

    f_x_test = util.output_in_batches(x_test, params, f_jit, FLAGS.batch_count_accuracy)
    f_x_lin_test = util.output_in_batches(x_test, params_lin, f_lin_jit, FLAGS.batch_count_accuracy)

    # Print out summary data comparing the linear / nonlinear model.
    util.print_summary('train', y_train, f_x, f_x_lin, loss)
    util.print_summary('test', y_test, f_x_test, f_x_lin_test, loss)


if __name__ == '__main__':
    app.run(main)
