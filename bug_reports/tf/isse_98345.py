import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)	# TensorFlow version: 2.21.0-dev20250801

rng = np.random.default_rng(957)

lower_upper = tf.constant(rng.uniform(-9223372036854772000., -9223372036854771000., size=(3, 3)), dtype=tf.float64)
perm = tf.constant(rng.uniform(0., 0., size=(3,)), dtype=tf.int64)
rhs = tf.constant(rng.uniform(-100., 100., size=(3, 3)), dtype=tf.float64)
validate_args = True
name = "constant"

with tf.device("/CPU:0"):
        result = tf.linalg.lu_solve(
            lower_upper=lower_upper,
            perm=perm,
            rhs=rhs,
            validate_args=validate_args,
            name=name
        )
        print("Result (CPU):\n", result.numpy()[0, 2])  # -0.0

with tf.device("/GPU:0"):
        result = tf.linalg.lu_solve(
            lower_upper=lower_upper,
            perm=perm,
            rhs=rhs,
            validate_args=validate_args,
            name=name
        )
        print("Result (GPU):\n", result.numpy()[0, 2])  #  -47201.82360354644