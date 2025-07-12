
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_map_flat_values_inputs():
    list_of_inputs = []

    # Input 1
    rt = tf.ragged.constant([[1, 2, 3], [], [4, 5], [6]])
    op = tf.ones_like
    input_dict = {"op": op, "*args": [rt]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    rt = tf.ragged.constant([[1, 2, 3], [], [4, 5], [6]])
    op = tf.math.multiply
    input_dict = {"op": op, "*args": [rt, rt]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    rt = tf.ragged.constant([[1, 2, 3], [], [4, 5], [6]])
    const_val = tf.constant(5)
    op = lambda x, const_val=const_val: tf.math.add(x, const_val)
    input_dict = {"op": op, "*args": [rt]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    rt = tf.ragged.constant([[-1, -2, -3], [], [-4, -5], [-6]])
    op = tf.abs
    input_dict = {"op": op, "*args": [rt]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    rt1 = tf.ragged.constant([[1, 2], [3, 4, 5]])
    rt2 = tf.ragged.constant([[6, 7], [8, 9, 10]])
    op = tf.math.add
    input_dict = {"op": op, "*args": [rt1, rt2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    rt = tf.ragged.constant([[[1, 2], [3]], [[4, 5, 6]]])
    op = tf.negative
    input_dict = {"op": op, "*args": [rt]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    rt = tf.ragged.constant([[1.0, 2.0], [], [3.0, 4.0, 5.0]])
    op = tf.math.sqrt
    input_dict = {"op": op, "*args": [rt]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    rt = tf.ragged.constant([[], [1, 2], [3]])
    op = tf.math.square
    input_dict = {"op": op, "*args": [rt]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    rt = tf.ragged.constant([[1, 2, 3], [4]])
    op = lambda x: tf.cast(x, tf.float32)
    input_dict = {"op": op, "*args": [rt]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    rt1 = tf.ragged.constant([[1, 2], [3, 4, 5]], dtype=tf.int64)
    rt2 = tf.ragged.constant([[6, 7], [8, 9, 10]], dtype=tf.int64)
    op = tf.math.floordiv
    input_dict = {"op": op, "*args": [rt1, rt2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ragged.map_flat_values"] = tf_ragged_map_flat_values_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ragged.map_flat_values' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.map_flat_values'.")

check_valid('tf.ragged.map_flat_values', generated_inputs['tf.ragged.map_flat_values'], lib="tf", suffix=0)
