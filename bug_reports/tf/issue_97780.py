import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)	# 2.21.0-dev20250729

rng = np.random.default_rng(395)

tensor = tf.constant(rng.uniform(-127, 80, (2, 8, 96, 62, 1, 1)), dtype=tf.float64)

eigenvalues_numpy, eigenvectors_numpy = np.linalg.eigh(tensor.numpy())
print(eigenvalues_numpy.shape, eigenvectors_numpy.shape) # no error

with tf.device('/GPU:0'):
	result_e, result_v = tf.linalg.eigh(tensor) # crash