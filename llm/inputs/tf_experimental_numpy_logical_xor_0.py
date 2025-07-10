
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_logical_xor_inputs():
    list_of_inputs = []

    # Input 1: Basic boolean tensors
    x1 = tf.constant([True, False, True, False])
    x2 = tf.constant([False, False, True, True])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Numerical tensors (0 and 1)
    x1 = tf.constant([0, 1, 0, 1], dtype=tf.int32)
    x2 = tf.constant([1, 1, 0, 0], dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tensors with different shapes, broadcastable
    x1 = tf.constant([[True, False], [False, True]])
    x2 = tf.constant([False, True], dtype=tf.bool)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger tensors, different shapes, broadcastable
    x1 = tf.constant([True, False, True])
    x2 = tf.constant([[False], [True], [False]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional tensors
    x1 = tf.constant([[[True, False], [False, True]], [[False, True], [True, False]]])
    x2 = tf.constant([[[False, True], [True, False]], [[True, False], [False, True]]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All False
    x1 = tf.constant([False, False, False])
    x2 = tf.constant([False, False, False])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All True
    x1 = tf.constant([True, True, True])
    x2 = tf.constant([True, True, True])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger tensors with mixed True/False
    x1 = tf.constant([[True, False, True], [False, True, False]])
    x2 = tf.constant([[False, True, False], [True, False, True]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Boolean tensors with different shapes
    x1 = tf.constant([True, False])
    x2 = tf.constant([[False], [True]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty tensors (avoiding size issue)
    x1 = tf.constant(np.array([]).astype(bool))
    x2 = tf.constant(np.array([]).astype(bool))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.logical_xor"] = tf_experimental_numpy_logical_xor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.logical_xor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.logical_xor'.")

check_valid('tf.experimental.numpy.logical_xor', generated_inputs['tf.experimental.numpy.logical_xor'], lib="tf", suffix=0)
