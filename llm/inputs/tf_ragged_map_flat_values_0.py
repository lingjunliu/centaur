
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_map_flat_values_inputs():
    list_of_inputs = []

    # Input 1: Simple case with tf.ones_like
    rt1 = tf.ragged.constant([[1, 2, 3], [], [4, 5], [6]])
    input_dict1 = {"op": tf.ones_like, "*args": [rt1]}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Multiplication of two ragged tensors
    rt2_1 = tf.ragged.constant([[1, 2, 3], [], [4, 5], [6]])
    rt2_2 = tf.ragged.constant([[7, 8, 9], [], [10, 11], [12]])
    input_dict2 = {"op": tf.multiply, "*args": [rt2_1, rt2_2]}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Adding a constant value
    rt3 = tf.ragged.constant([[1, 2, 3], [], [4, 5], [6]])
    input_dict3 = {"op": tf.add, "*args": [rt3, tf.constant(5)]}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Ragged tensor with different data type
    rt4 = tf.ragged.constant([[1.0, 2.0, 3.0], [], [4.0, 5.0], [6.0]])
    input_dict4 = {"op": tf.negative, "*args": [rt4]}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Ragged tensor with more nested levels
    rt5 = tf.ragged.constant([[[1, 2], [3]], [[4, 5, 6], []]])
    input_dict5 = {"op": tf.square, "*args": [rt5]}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Two ragged tensors with deeper nesting
    rt6_1 = tf.ragged.constant([[[1, 2], [3]], [[4, 5, 6], []]])
    rt6_2 = tf.ragged.constant([[[7, 8], [9]], [[10, 11, 12], []]])
    input_dict6 = {"op": tf.add, "*args": [rt6_1, rt6_2]}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7:  Using tf.math.floordiv directly
    rt7 = tf.ragged.constant([[7, 8, 9], [], [10, 11], [12]])
    input_dict7 = {"op": tf.math.floordiv, "*args": [rt7, tf.constant(3)]}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Boolean ragged tensor
    rt8 = tf.ragged.constant([[True, False, True], [], [False, True], [True]])
    input_dict8 = {"op": tf.logical_not, "*args": [rt8]}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Identity operation
    rt9 = tf.ragged.constant([[1, 2, 3], [], [4, 5], [6]])
    input_dict9 = {"op": tf.identity, "*args": [rt9]}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: tf.cast
    rt10 = tf.ragged.constant([[1.0, 2.0, 3.0], [], [4.0, 5.0], [6.0]])
    input_dict10 = {"op": lambda x: tf.cast(x, tf.int32), "*args": [rt10]}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: tf.sin
    rt11 = tf.ragged.constant([[1.0, 2.0, 3.0], [], [4.0, 5.0], [6.0]])
    input_dict11 = {"op": tf.sin, "*args": [rt11]}
    list_of_inputs.append(copy.deepcopy(input_dict11))

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
