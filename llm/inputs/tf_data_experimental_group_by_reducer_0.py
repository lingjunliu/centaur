
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
    input_dict = {'key_func': key_func, 'reducer': reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    key_func = [lambda x: tf.cast(tf.strings.length(x), tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant("", dtype=tf.string),
        reduce_func=lambda x, y: tf.strings.join([x, y], separator=","),
        finalize_func=lambda x: x
    )]
    input_dict = {'key_func': key_func, 'reducer': reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    key_func = [lambda x: tf.cast(tf.shape(x)[0], tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(0, dtype=tf.float32),
        reduce_func=lambda x, y: x + tf.reduce_sum(tf.cast(y, dtype=tf.float32)),
        finalize_func=lambda x: x
    )]
    input_dict = {'key_func': key_func, 'reducer': reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    key_func = [lambda x: tf.cast(tf.reduce_sum(x), tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant([], dtype=tf.int32),
        reduce_func=lambda x, y: tf.concat([x, tf.reshape(y, [-1])], axis=0),
        finalize_func=lambda x: x
    )]
    input_dict = {'key_func': key_func, 'reducer': reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    key_func = [lambda x: tf.cast(tf.cond(tf.reduce_all(x > 0), lambda: 1, lambda: 0), tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(True, dtype=tf.bool),
        reduce_func=lambda x, y: tf.logical_and(x, tf.reduce_all(y > 0)),
        finalize_func=lambda x: x
    )]
    input_dict = {'key_func': key_func, 'reducer': reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    key_func = [lambda x: tf.cast(tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32), tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(0, dtype=tf.int32),
        reduce_func=lambda x, y: x + 1,
        finalize_func=lambda x: tf.cast(x, tf.float32)
    )]
    input_dict = {'key_func': key_func, 'reducer': reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    key_func = [lambda x: tf.cast(tf.size(x), tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(0, dtype=tf.int64),
        reduce_func=lambda x, y: x + tf.cast(tf.size(y), tf.int64),
        finalize_func=lambda x: x
    )]
    input_dict = {'key_func': key_func, 'reducer': reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    key_func = [lambda x: tf.cast(tf.math.floormod(tf.cast(x, tf.float32), 5.0), tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(0.0, dtype=tf.float32),
        reduce_func=lambda x, y: x + tf.cast(y, tf.float32),
        finalize_func=lambda x: x
    )]
    input_dict = {'key_func': key_func, 'reducer': reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    key_func = [lambda x: tf.cast(x[0], tf.int64)] #Simplified lambda
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(0, dtype=tf.int64),
        reduce_func=lambda x, y: x + tf.cast(y[1], tf.int64),
        finalize_func=lambda x: x
    )]
    input_dict = {'key_func': key_func, 'reducer': reducer}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    key_func = [lambda x: tf.cast(tf.size(x)%3, tf.int64)]
    reducer = [DummyReducer(
        init_func=lambda: tf.constant(0, dtype=tf.int64),
        reduce_func=lambda x, y: x + tf.cast(tf.size(y), tf.int64),
        finalize_func=lambda x: x
    )]
    input_dict = {'key_func': key_func, 'reducer': reducer}
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
