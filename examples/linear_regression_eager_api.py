import os
import numpy
import tensorflow as tf
import numpy.random as random
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set Eager API
tf.enable_eager_execution()
tfe = tf.contrib.eager

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_x = numpy.asarray(
    [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_y = numpy.asarray(
    [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n = len(train_x)

# Set model weights
W = tf.Variable(random.randn())
b = tf.Variable(random.randn())


# Linear regression (Wx + b)
def linear_regression(inputs):
    return inputs * W + b


# Mean square error
def mean_square_fn(model_fn, inputs, labels):
    return tf.reduce_sum(tf.pow(model_fn(inputs) - labels, 2)) / n


# SGD Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Compute gradients
grad = tfe.implicit_gradients(mean_square_fn)

# Initial cost, before optimizing
print("Initial cost= {:.9f}".format(
    mean_square_fn(linear_regression, train_x, train_y)),
    "W=", W.numpy(), "b=", b.numpy())

# Training
for step in range(training_epochs):

    optimizer.apply_gradients(grad(linear_regression, train_x, train_y))

    if (step + 1) % display_step == 0 or step == 0:
        print("Epoch:", '%04d' % (step + 1), "cost=",
              "{:.9f}".format(mean_square_fn(linear_regression, train_x, train_y)),
              "W=", W.numpy(), "b=", b.numpy())

# Graphic display
plt.plot(train_x, train_y, 'ro', label='Original data')
plt.plot(train_x, numpy.array(W * train_x + b), label='Fitted line')
plt.legend()
plt.show()
