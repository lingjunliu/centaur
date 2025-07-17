
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_group_by_reducer_inputs():
    list_of_inputs = []

    class DummyReducer:
        def __init__(self, init_func, reduce_func, finalize_func):
            self.init_func = init_func
            self.reduce_func = reduce_func
            self.finalize_func = finalize_func

    # Input 1
    key_func = [lambda x: tf.cast(x % 2, tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(0, dtype=tf.int64),
        reduce_func=lambda x, y: x + tf.cast(y, tf.int64),
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    key_func = [lambda x: tf.cast(tf.strings.length(x), tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant("", dtype=tf.string),
        reduce_func=lambda x, y: tf.strings.join([x, y], separator=","),
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    key_func = [lambda x: tf.cast(tf.shape(x)[0], tf.int64) if tf.rank(x) > 0 else tf.constant(0, dtype=tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant([], dtype=tf.float32),
        reduce_func=lambda x, y: tf.concat([x, tf.reshape(tf.cast(y, tf.float32), [-1])], axis=0) if tf.rank(y) > 0 else x,
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    key_func = [lambda x: tf.cast(tf.reduce_sum(x), tf.int64) if tf.rank(x) > 0 else tf.constant(0, dtype=tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(0.0, dtype=tf.float32),
        reduce_func=lambda x, y: x + tf.cast(tf.reduce_sum(y), tf.float32) if tf.rank(y) > 0 else x,
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    key_func = [lambda x: tf.cast(tf.reduce_prod(x), tf.int64) if tf.rank(x) > 0 else tf.constant(1, dtype=tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(1, dtype=tf.int64),
        reduce_func=lambda x, y: x * tf.cast(tf.reduce_prod(y), tf.int64) if tf.rank(y) > 0 else x,
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    key_func = [lambda x: tf.cast(tf.math.count_nonzero(x), tf.int64) if tf.rank(x) > 0 else tf.constant(0, dtype=tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(0, dtype=tf.int64),
        reduce_func=lambda x, y: x + tf.cast(tf.math.count_nonzero(y), tf.int64) if tf.rank(y) > 0 else x,
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    key_func = [lambda x: tf.cast(x[0], tf.int64) if tf.rank(x) > 0 and tf.size(x) > 0 else tf.constant(0, dtype=tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(0, dtype=tf.int64),
        reduce_func=lambda x, y: x + tf.cast(y[1], tf.int64) if tf.rank(y) > 0 and tf.size(y) > 1 else x,
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    key_func = [lambda x: tf.cast(tf.size(x), tf.int64) if tf.rank(x) > 0 else tf.constant(0, dtype=tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(0, dtype=tf.int64),
        reduce_func=lambda x, y: x + tf.cast(tf.size(y), tf.int64) if tf.rank(y) > 0 else x,
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    key_func = [lambda x: tf.cast(tf.cond(tf.reduce_sum(x) > 0, lambda: 1, lambda: 0), tf.int64) if tf.rank(x) > 0 else tf.constant(0, dtype=tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(0, dtype=tf.int64),
        reduce_func=lambda x, y: x + tf.cast(tf.size(y), tf.int64) if tf.rank(y) > 0 else x,
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    key_func = [lambda x: tf.cast(tf.math.floormod(tf.cast(tf.strings.length(x), tf.int64), 3), tf.int64) if isinstance(x, str) else tf.constant(0, dtype=tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant("", dtype=tf.string),
        reduce_func=lambda x, y: tf.strings.join([x, y], separator="") if isinstance(y, str) else x,
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    key_func = [lambda x: tf.cast(x % 5, tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(0, dtype=tf.int64),
        reduce_func=lambda x, y: x + tf.cast(y, tf.int64),
        finalize_func=lambda x: x
    )]
    input_dict = {"key_func": key_func, "reducer": reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    key_func = [lambda x: tf.cast(tf.math.floormod(x, 3), tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(0, dtype=tf.int64),
        reduce_func=lambda x, y: x + tf.cast(y, tf.int64),
        finalize_func=lambda x: x
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
    
    print("Valid")

if 'tf.data.experimental.group_by_reducer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.group_by_reducer'.")

check_valid('tf.data.experimental.group_by_reducer', generated_inputs['tf.data.experimental.group_by_reducer'], lib="tf", suffix=0)
