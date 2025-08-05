import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)	# TensorFlow version: 2.21.0-dev20250801

rng = np.random.default_rng(253)

diagonal = tf.constant(rng.uniform(0, 1, size=(9, 3, 70, 59, 22, 30)), dtype=tf.float32)

with tf.device("/GPU:0"):
    output = tf.raw_ops.MatrixDiag(diagonal=diagonal)