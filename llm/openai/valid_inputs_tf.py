generated_inputs = {}

import tensorflow as tf
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
import copy