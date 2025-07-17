# tf.math.argmax
import tensorflow as tf
import numpy as np

print(tf.__version__)   # 2.19.0
rng = np.random.default_rng(311)

input_tensor = tf.constant(rng.uniform(-np.finfo(np.float16).max-1, 0., size=(1, 2)), dtype=tf.float16)
axis = tf.constant(0, dtype=tf.int16)
output_type = tf.int16

output = tf.math.argmax(input_tensor, axis=axis, output_type=output_type)
# InvalidArgumentError: {{function_node __wrapped__ArgMax_device_/job:localhost/replica:0/task:0/device:CPU:0}} Expected dimension in the range [-2, 2), but got -991297536 [Op:ArgMax] name:

# tf.experimental.numpy.argmax

rng = np.random.default_rng(288)

input_tensor = tf.constant(rng.uniform(-1., 0., size=(18, 1, 2, 2)), dtype=tf.int16)
axis = tf.constant(0, dtype=tf.int16)

output = tf.experimental.numpy.argmax(input_tensor, axis=axis)
# tensorflow.python.framework.errors_impl.InvalidArgumentError: {{function_node __wrapped__ArgMax_device_/job:localhost/replica:0/task:0/device:CPU:0}} Expected dimension in the range [-4, 4), but got 1450967040 [Op:ArgMax] name:
