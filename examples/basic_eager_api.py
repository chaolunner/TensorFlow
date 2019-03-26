# TensorFlow's eager execution is an imperative programming environment that evaluates operations immediately,
# without building graphs: operations return concrete values instead of constructing a computational graph to run later.
# This makes it easy to get started with TensorFlow and debug models, and it reduces boilerplate as well.
# To follow along with this guide, run the code samples below in an interactive python interpreter.
# https://www.tensorflow.org/guide/eager

import tensorflow.contrib.eager as tfe
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Setting Eager mode...")
tfe.enable_eager_execution()

print("Define constant tensors")
a = tf.constant(2)
print("a = %i" % a)
b = tf.constant(3)
print("b = %i" % b)

print("Running operations, without tf.Session")
c = a + b
print("a + b = %i" % c)
d = a * b
print("a * b = %i" % d)

print("Mixing operations with Tensors and Numpy Arrays")
a = tf.constant([[2., 1.], [1., 0.]], dtype=tf.float32)
print("Tensor:\n a = %s" % a)
b = np.array([[3., 0.], [5., 1.]], dtype=np.float32)
print("NumpyArray:\n b = %s" % b)

print("Running operations, without tf.Session")
c = a + b
print("a + b = %s" % c)
d = tf.matmul(a, b)
print("a * b = %s" % d)

print("Iterate through Tensor 'a':")
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])
