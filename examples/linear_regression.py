import os
import numpy
import tensorflow as tf
import numpy.random as random
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_x = numpy.asarray(
    [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_y = numpy.asarray(
    [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n = train_x.shape[0]

# tf Graph Input
x = tf.placeholder("float")
y = tf.placeholder("float")

# Set model weights
W = tf.Variable(random.randn(), name="weight")
b = tf.Variable(random.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(x, W), b)

# Mean squared error, MSE
cost = tf.reduce_sum(tf.pow(pred - y, 2)) / n
# Gradient descent
# Note, minimize() knows to modify w and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (item_x, item_y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={x: item_x, y: item_y})

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={x: train_x, y: train_y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={x: train_x, y: train_y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example
    test_x = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(tf.reduce_sum(tf.pow(pred - y, 2)) / test_x.shape[0],
                            feed_dict={x: test_x, y: test_y})  # same function as cost above
    print("Testing cost=", testing_cost, "W=", sess.run(W), "b=", sess.run(b))
    print("Absolute mean square loss difference:", abs(training_cost - testing_cost))

    plt.plot(test_x, test_y, 'bo', label='Testing data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
