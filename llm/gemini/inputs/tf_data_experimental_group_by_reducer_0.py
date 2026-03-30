
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_data_experimental_group_by_reducer_inputs():
    list_of_inputs = []

    class DummyReducer:
        def __init__(self, init_func, reduce_func, finalize_func):
            self.init_func = init_func
            self.reduce_func = reduce_func
            self.finalize_func = finalize_func

    def create_reducer(init_func, reduce_func, finalize_func):
        class MyReducer:
            def __init__(self):
                self.init_func = init_func
                self.reduce_func = reduce_func
                self.finalize_func = finalize_func
        return MyReducer()

    # Input 1
    key_func = [lambda x: tf.cast(x % 2, tf.int64)]
    reducer = [create_reducer(
        init_func=lambda: tf.constant(0, dtype=tf.int64),
        reduce_func=lambda x, y: x + tf.cast(y, tf.int64),
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    key_func = [lambda x: tf.cast(x // 5, tf.int64)]
    reducer = [create_reducer(
        init_func=lambda: tf.constant(1, dtype=tf.int64),
        reduce_func=lambda x, y: x * tf.cast(y, tf.int64),
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    key_func = [lambda x: tf.cast(tf.strings.length(x), tf.int64)]
    reducer = [create_reducer(
        init_func=lambda: tf.constant("", dtype=tf.string),
        reduce_func=lambda x, y: tf.strings.join([x, y]),
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    key_func = [lambda x: tf.cast(tf.shape(x)[0], tf.int64)]
    reducer = [create_reducer(
        init_func=lambda: tf.constant(0, dtype=tf.float32),
        reduce_func=lambda x, y: x + tf.reduce_sum(tf.cast(y, tf.float32)),
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    def key_func_5(x):
      return tf.cast(tf.reduce_sum(x), tf.int64)
    key_func = [key_func_5]
    reducer = [create_reducer(
        init_func=lambda: tf.constant(0, dtype=tf.int64),
        reduce_func=lambda x, y: x + tf.cast(tf.reduce_sum(y), tf.int64),
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    def key_func_6(x):
      return tf.cast(tf.reduce_sum(tf.cast(tf.math.is_finite(x), tf.int64)), tf.int64)
    key_func = [key_func_6]
    reducer = [create_reducer(
        init_func=lambda: tf.constant(0, dtype=tf.int64),
        reduce_func=lambda x, y: x + tf.cast(tf.reduce_sum(tf.cast(tf.math.is_finite(y), tf.int64)), tf.int64),
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    def key_func_7(x):
      return tf.cast(tf.shape(x)[-1], tf.int64)
    key_func = [key_func_7]
    reducer = [create_reducer(
        init_func=lambda: tf.constant(1.0, dtype=tf.float64),
        reduce_func=lambda x, y: x * tf.cast(tf.reduce_sum(y), tf.float64),
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    def key_func_8(x):
      return tf.cast(tf.reduce_max(x), tf.int64)
    key_func = [key_func_8]
    reducer = [create_reducer(
        init_func=lambda: tf.constant(0, dtype=tf.int32),
        reduce_func=lambda x, y: x + tf.cast(tf.reduce_min(y), tf.int32),
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    def key_func_9(x):
      return tf.cast(tf.size(x), tf.int64)
    key_func = [key_func_9]
    reducer = [create_reducer(
        init_func=lambda: tf.constant(True, dtype=tf.bool),
        reduce_func=lambda x, y: tf.logical_and(x, tf.reduce_all(y)),
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    def key_func_10(x):
      return tf.cast(tf.random.uniform(shape=[], minval=0, maxval=10, dtype=tf.int32), tf.int64)
    key_func = [key_func_10]
    reducer = [create_reducer(
        init_func=lambda: tf.constant([0, 0, 0], dtype=tf.int32),
        reduce_func=lambda x, y: x + tf.cast([tf.reduce_sum(y), tf.reduce_mean(y), tf.reduce_max(y)], dtype=tf.int32),
        finalize_func=lambda x: tf.cast(x, tf.float32)
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.group_by_reducer"] = tf_data_experimental_group_by_reducer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.group_by_reducer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.group_by_reducer'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.group_by_reducer', generated_inputs['tf.data.experimental.group_by_reducer'], lib="tf", suffix=0)
