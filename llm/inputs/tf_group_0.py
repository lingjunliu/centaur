
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_group_inputs():
    list_of_inputs = []

    # Input 1: Empty list of tensors
    input_dict = {
        "inputs": [],
        "name": "empty_group"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single tensor
    a = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    input_dict = {
        "inputs": [a],
        "name": "single_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Two 1D tensors of same dtype but different values
    a = tf.constant(np.array([1, 2], dtype=np.int32))
    b = tf.constant(np.array([3, 4], dtype=np.int32))
    input_dict = {
        "inputs": [a, b],
        "name": "two_1d_tensors"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensors with different data types, same shape
    a = tf.constant(np.array([1, 2], dtype=np.int32))
    b = tf.constant(np.array([4.0, 5.0], dtype=np.float32))
    input_dict = {
        "inputs": [a, b],
        "name": "tensors_diff_dtypes"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar tensor and a 1D tensor
    a = tf.constant(np.array(10, dtype=np.int32))
    b = tf.constant(np.array([3.14, 2.71], dtype=np.float32))
    input_dict = {
        "inputs": [a, b],
        "name": "scalar_tensors"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensors with negative values
    a = tf.constant(np.array([-1, -2], dtype=np.int32))
    b = tf.constant(np.array([-4.5, -5.6], dtype=np.float32))
    input_dict = {
        "inputs": [a, b],
        "name": "negative_values"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensors with zero values
    a = tf.constant(np.array([0, 0], dtype=np.int32))
    b = tf.constant(np.array([0.0, 0.0], dtype=np.float32))
    input_dict = {
        "inputs": [a, b],
        "name": "zero_values"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  Two larger tensors of same shape
    a = tf.constant(np.random.rand(5, 5).astype(np.float32))
    b = tf.constant(np.random.rand(5, 5).astype(np.int32))
    input_dict = {
        "inputs": [a, b],
        "name": "larger_tensors"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  Three 1D tensors
    a = tf.constant(np.array([1], dtype=np.int32))
    b = tf.constant(np.array([2], dtype=np.int32))
    c = tf.constant(np.array([3], dtype=np.int32))
    input_dict = {
        "inputs": [a, b, c],
        "name": "more_tensors"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: Single 2D Tensor
    a = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    input_dict = {
        "inputs": [a],
        "name": "single_2d_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.group"] = tf_group_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.group' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.group'.")

check_valid('tf.group', generated_inputs['tf.group'], lib="tf", suffix=0)
