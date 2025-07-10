
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.experimental.numpy.experimental_enable_numpy_behavior()

def tf_experimental_numpy_isposinf_inputs():
    list_of_inputs = []

    # Input 1: Basic positive infinity
    x = tf.constant([np.inf, -np.inf, 0.0, 1.0], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Array of positive and negative infinities
    x = tf.constant(np.array([[np.inf, -np.inf], [np.nan, 1.0]]), dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: All positive infinities
    x = tf.constant(np.array([np.inf, np.inf, np.inf]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed values including positive infinity
    x = tf.constant(np.array([1.0, np.inf, 2.0, -np.inf, np.nan]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional array
    x = tf.constant(np.array([[[np.inf, 1.0], [2.0, -np.inf]], [[np.nan, 3.0], [4.0, np.inf]]]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All zeros
    x = tf.constant(np.array([0.0, 0.0, 0.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Large tensor
    x = tf.constant(np.full((10, 10), np.inf), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with multiple data types, convert to float32
    x = tf.constant([1.0, 2.0, 3.0, np.inf], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with different float type
    x = tf.constant(np.array([np.inf, -np.inf, 0.0, 1.0]), dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty Tensor
    x = tf.constant([], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Reshaped
    x = tf.reshape(tf.constant(np.array([np.inf, 1.0, -np.inf, 2.0, np.nan, 3.0]), dtype=tf.float32), (2, 3))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.isposinf"] = tf_experimental_numpy_isposinf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.isposinf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isposinf'.")

check_valid('tf.experimental.numpy.isposinf', generated_inputs['tf.experimental.numpy.isposinf'], lib="tf", suffix=0)
