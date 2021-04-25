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
# from jax.experimental.stax import logsoftmax
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
import datasets
import util

import pdb

flags.DEFINE_float('learning_rate', 0.1,
                   'Learning rate to use during training.')
flags.DEFINE_integer('batch_size', 10,
                     'Batch size to use during training.')
flags.DEFINE_integer('batch_size_kernel', 0,
                     'Batch size for kernel construction, 0 for no batching.')
flags.DEFINE_integer('train_epochs', 10,
                     'Number of epochs to train for.')
flags.DEFINE_integer('network_width', 1,
                     'Factor by which the network width is multiplied.')
flags.DEFINE_integer('learning_decline', 5000,
                     'Number of epochs after which the learning rate is divided by 10.')
flags.DEFINE_integer('batch_count_accuracy', 10,
                     'Number of batches when computing output over entire data set.')

FLAGS = flags.FLAGS


def main(unused_argv):
    # Load and normalize data
    print('Loading data...')
    x_train, y_train, x_test, y_test = datasets.get_dataset('mnist', n_train=10, n_test=10,
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
    print(f'Network of width x{FLAGS.network_width} built.')

    # Construct the kernel function
    kernel_fn = nt.batch(kernel_fn, device_count=-1, batch_size=FLAGS.batch_size_kernel)
    print('Kernel constructed')

    # Compute random initial parameters
    key = random.PRNGKey(0)
    _, params = init_fn(key, (-1, 28, 28, 1))

    # Linearize the network about its initial parameters.
    f_lin = nt.linearize(f, params)

    # Create a callable function for dynamic learning rates
    # Starts with learning_rate, divided by 10 after learning_decline epochs.
    dynamic_learning_rate = lambda iteration_step: FLAGS.learning_rate / 10**((iteration_step //
                                                                              (x_train.shape[0] // FLAGS.batch_size)) //
                                                                              FLAGS.learning_decline)

    # Create and initialize an optimizer for both f and f_lin.
    opt_init, opt_apply, get_params = optimizers.momentum(dynamic_learning_rate, 0.9)
    opt_apply = jit(opt_apply)

    # Compute the initial states
    state = opt_init(params)
    state_lin = opt_init(params)

    # # Create a cross-entropy loss function.
    # loss = lambda fx, y_hat: -np.mean(logsoftmax(fx) * y_hat)

    # Define the accuracy function
    accuracy = lambda fx, y_hat: np.mean(np.argmax(fx, axis=1) == np.argmax(y_hat, axis=1))

    # Define mean square error loss function
    loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)

    # Specialize the loss function to compute gradients for both linearized and
    # full networks.
    grad_loss = jit(grad(lambda params, x, y: loss(f(params, x), y)))
    grad_loss_lin = jit(grad(lambda params, x, y: loss(f_lin(params, x), y)))

    # Train the network.
    print('Training...')
    print('Epoch\tTime\tAccuracy\tLin. Accuracy\tLoss\tLin. Loss')
    print('-----------------------------------------------------------------')

    # Initialize training
    epoch = 1
    steps_per_epoch = x_train.shape[0] // FLAGS.batch_size

    start = time.time()
    start_epoch = time.time()

    accuracy_sum = 0
    lin_accuracy_sum = 0

    # accuracy_array = [0] * steps_per_epoch
    # lin_accuracy_array = [0] * steps_per_epoch
    # loss_array = [0] * steps_per_epoch
    # lin_loss_array = [0] * steps_per_epoch

    for i, (x, y) in enumerate(datasets.minibatch(
            x_train, y_train, FLAGS.batch_size, FLAGS.train_epochs)):

        # Update the parameters
        params = get_params(state)

        import pdb
        pdb.set_trace()
        breakpoint()
        # /scicore/scratch/kesben00/

        state = opt_apply(i, grad_loss(params, x, y), state)

        params_lin = get_params(state_lin)
        state_lin = opt_apply(i, grad_loss_lin(params_lin, x, y), state_lin)

        # TO-DO: Print exact accuracy every n (=100?) epochs
        accuracy_sum += accuracy(f(params, x), y)
        lin_accuracy_sum += accuracy(f_lin(params_lin, x), y)

        # # Save the accuracy and loss in an array
        # pointer = i % steps_per_epoch
        # accuracy_array[pointer] = float(accuracy(f(params, x), y))
        # lin_accuracy_array[pointer] = float(accuracy(f_lin(params_lin, x), y))
        #
        # loss_array[pointer] = float(loss(f(params, x), y))
        # lin_loss_array[pointer] = float(loss(f_lin(params_lin, x), y))

        # End of epoch
        if (i + 1) % steps_per_epoch == 0:
            time_point = time.time() - start_epoch
            # Print information about past epoch

            # accuracy_tmp = np.array(accuracy_array)
            # lin_accuracy_tmp = np.array(lin_accuracy_array)
            # loss_tmp = np.array(loss_array)
            # lin_loss_tmp = np.array(lin_loss_array)
            # print('{}\t{:.3f}\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t{:.4f}'.format(
            #     epoch, time_point, np.mean(accuracy_tmp) * 100, np.mean(lin_accuracy_tmp) * 100,
            #     np.mean(loss_tmp), np.mean(lin_loss_tmp)))

            print('{}\t{:.3f}\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t{:.4f}'.format(
                epoch, time_point, accuracy_sum / steps_per_epoch * 100, lin_accuracy_sum / steps_per_epoch * 100,
                loss(f(params, x), y), loss(f_lin(params_lin, x), y)))
            accuracy_sum = 0
            lin_accuracy_sum = 0

            # print('{}\t{:.3f}\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t{:.4f}'.format(
            #     epoch, time_point, accuracy(f(params, x), y) * 100, accuracy(f_lin(params_lin, x), y) * 100,
            #     loss(f(params, x), y), loss(f_lin(params_lin, x), y)))

            epoch += 1
            # Reset timer
            start_epoch = time.time()

    duration = time.time() - start
    print('-----------------------------------------------------------------')
    print(f'Training complete in {duration} seconds.')

    # x, y = x_train[:10000], y_train[:10000]
    # x, y = x_train, y_train

    # f_x = []
    # f_x_lin = []
    # train_set_batch = x_train.shape[0] / FLAGS.batch_count_accuracy
    # # Compute output on train set in batches
    # for i in range(FLAGS.batch_count_accuracy):
    #     start = int(i*train_set_batch)
    #     end = int((i+1)*train_set_batch)
    #     x, y = x_train[start:end], y_train[start:end]
    #     f_x.extend(f(params, x))
    #     f_x_lin.extend(f_lin(params_lin, x))
    # f_x = np.array(f_x)
    # f_x_lin = np.array(f_x_lin)

    # Print out summary data comparing the linear / nonlinear model.

    # Compute output in batches
    f_x = util.output_in_batches(x_train, y_train, params, f, FLAGS.batch_count_accuracy)
    f_x_lin = util.output_in_batches(x_train, y_train, params_lin, f_lin, FLAGS.batch_count_accuracy)

    f_x_test = util.output_in_batches(x_test, y_test, params, f, FLAGS.batch_count_accuracy//5)
    f_x_lin_test = util.output_in_batches(x_test, y_test, params_lin, f_lin, FLAGS.batch_count_accuracy//5)

    util.print_summary('train', y_train, f_x, f_x_lin, loss)
    # util.print_summary(
    #     'test', y_test, f(params, x_test), f_lin(params_lin, x_test), loss)
    util.print_summary('compare', y_test, f_x_test, f_x_lin_test, loss)


if __name__ == '__main__':
    app.run(main)
