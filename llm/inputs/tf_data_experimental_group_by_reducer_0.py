
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_group_by_reducer_inputs():
    list_of_inputs = []

    # Input 1
    def key_func1(x):
        return tf.cast(x % 2, tf.int64)

    class Reducer1:
        def __init__(self):
            pass

        def init_func(self, key):
            return tf.constant(0, dtype=tf.int64)

        def reduce_func(self, state, element):
            return state + tf.cast(element, tf.int64)

        def finalize_func(self, state):
            return state

    reducer1 = Reducer1()
    input_dict = {"key_func": [key_func1], "reducer": [reducer1]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    def key_func2(x):
        return tf.cast(x % 3, tf.int64)

    class Reducer2:
        def __init__(self):
            pass

        def init_func(self, key):
            return tf.constant(0, dtype=tf.int64)

        def reduce_func(self, state, element):
            return state + tf.cast(element, tf.int64)

        def finalize_func(self, state):
            return state

    reducer2 = Reducer2()
    input_dict = {"key_func": [key_func2], "reducer": [reducer2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.data.experimental.group_by_reducer"] = tf_data_experimental_group_by_reducer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.group_by_reducer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.group_by_reducer'.")

check_valid('tf.data.experimental.group_by_reducer', generated_inputs['tf.data.experimental.group_by_reducer'], lib="tf", suffix=0)
