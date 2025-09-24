
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_isfinite_inputs():
    list_of_inputs = []

    # Input 1: Basic finite tensor
    x = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tensor with some infinities
    x = tf.constant(np.array([1.0, np.inf, 3.0, -np.inf]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tensor with some NaNs
    x = tf.constant(np.array([1.0, np.nan, 3.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with integers (should still work, converting to float)
    x = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multidimensional tensor
    x = tf.constant(np.array([[1.0, 2.0], [3.0, np.inf]]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All infs and nans
    x = tf.constant(np.array([np.inf, -np.inf, np.nan]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative finite numbers
    x = tf.constant(np.array([-1.0, -2.0, -3.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero values
    x = tf.constant(np.array([0.0, -0.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large numbers
    x = tf.constant(np.array([1e10, -1e10]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty array
    x = tf.constant(np.array([]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.isfinite"] = tf_experimental_numpy_isfinite_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.isfinite' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isfinite'.")

check_valid('tf.experimental.numpy.isfinite', generated_inputs['tf.experimental.numpy.isfinite'], lib="tf", suffix=0)
