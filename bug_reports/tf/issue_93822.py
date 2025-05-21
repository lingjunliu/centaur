"""
2: gpu_crash
SIGABRT

Re running the oracle...

Abstract input (seed 24):
input: 
        shape: (8, 10, 13, 12, 2, 4)
        dtype: <class 'numpy.float64'>
        range: (-10.9999152216621, 6.999802457840147)
rcond: 
        value: -2.0
        dtype: <class 'numpy.float64'>
"""
import numpy as np
import tensorflow as tf

rng = np.random.default_rng(24)

with tf.device('/GPU:0'):
    input_tensor = tf.constant(rng.uniform(-10.9999152216621, 6.999802457840147,(8, 10, 13, 12, 2, 4)), dtype=tf.float64)
    pinverse_tensor = tf.linalg.pinv(input_tensor)