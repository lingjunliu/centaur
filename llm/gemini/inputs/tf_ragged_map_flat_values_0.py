
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_ragged_map_flat_values_inputs():
    list_of_inputs = []

    # Input 1: Simple case with ones_like
    rt1 = tf.ragged.constant([[1, 2, 3], [], [4, 5]])
    op1 = tf.ones_like
    input_dict1 = {"op": op1, "*args": [rt1]}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Elementwise multiplication
    rt2 = tf.ragged.constant([[1, 2], [3, 4, 5], [6]])
    op2 = tf.multiply
    input_dict2 = {"op": op2, "*args": [rt2, rt2]}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Adding a constant
    rt3 = tf.ragged.constant([[1, 2, 3], [], [4, 5]])
    op3 = lambda x, y: tf.add(x, y)
    constant_tensor = tf.constant(5)
    input_dict3 = {"op": op3, "*args": [rt3, constant_tensor]}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: RaggedTensor with different shapes
    rt5 = tf.ragged.constant([[], [1], [2, 3, 4], [5, 6]])
    op5 = tf.negative
    input_dict5 = {"op": op5, "*args": [rt5]}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 5: RaggedTensor with negative values
    rt6 = tf.ragged.constant([[-1, -2], [-3, -4, -5], [-6]])
    op6 = tf.abs
    input_dict6 = {"op": op6, "*args": [rt6]}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 6: Using tf.math.sqrt
    rt7 = tf.ragged.constant([[1.0, 4.0], [9.0, 16.0]])
    op7 = tf.math.sqrt
    input_dict7 = {"op": op7, "*args": [rt7]}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 7: Two RaggedTensors with identical nested_row_splits
    rt8_1 = tf.ragged.constant([[1, 2], [3, 4, 5]])
    rt8_2 = tf.ragged.constant([[6, 7], [8, 9, 10]])
    op8 = tf.add
    input_dict8 = {"op": op8, "*args": [rt8_1, rt8_2]}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 8: Zeros like
    rt9 = tf.ragged.constant([[1, 2, 3], [], [4, 5]])
    op9 = tf.zeros_like
    input_dict9 = {"op": op9, "*args": [rt9]}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 9: Ragged tensor with float values and reciprocal
    rt10 = tf.ragged.constant([[2.0, 4.0], [6.0]])
    op10 = tf.math.reciprocal
    input_dict10 = {"op": op10, "*args": [rt10]}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ragged.map_flat_values"] = tf_ragged_map_flat_values_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.ragged.map_flat_values' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.map_flat_values'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.ragged.map_flat_values', generated_inputs['tf.ragged.map_flat_values'], lib="tf", suffix=0)
