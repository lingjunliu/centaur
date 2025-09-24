
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_sqrt_inputs():
    list_of_inputs = []

    # Input 1: Basic positive tensor
    x = tf.constant(np.array([1.0, 4.0, 9.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tensor with zeros
    x = tf.constant(np.array([0.0, 0.0, 0.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Higher dimensional tensor
    x = tf.constant(np.array([[1.0, 4.0], [9.0, 16.0]]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with mixed values
    x = tf.constant(np.array([0.0, 1.0, 2.0, 3.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with negative values (will produce NaN)
    x = tf.constant(np.array([-1.0, -4.0, -9.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  Double precision tensor
    x = tf.constant(np.array([1.0, 4.0, 9.0]), dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor
    x = tf.constant(np.array([[[1.0, 4.0], [9.0, 16.0]], [[25.0, 36.0], [49.0, 64.0]]]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with large values
    x = tf.constant(np.array([1e6, 1e8, 1e10]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with small values
    x = tf.constant(np.array([1e-6, 1e-8, 1e-10]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with mixed positive and negative and zero values
    x = tf.constant(np.array([-1.0, 0.0, 1.0, 4.0, -9.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.sqrt"] = tf_experimental_numpy_sqrt_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.sqrt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.sqrt'.")

check_valid('tf.experimental.numpy.sqrt', generated_inputs['tf.experimental.numpy.sqrt'], lib="tf", suffix=0)
