
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_isinf_inputs():
    list_of_inputs = []

    # Input 1: Scalar positive infinity
    x = tf.constant(np.inf, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar negative infinity
    x = tf.constant(-np.inf, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar finite number
    x = tf.constant(5.0, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with infinity and finite numbers
    x = tf.constant(np.array([np.inf, 1.0, -np.inf, 0.0], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with infinity and finite numbers
    x = tf.constant(np.array([[np.inf, 1.0], [-np.inf, 0.0]], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array with infinity and finite numbers
    x = tf.constant(np.array([[[np.inf, 1.0], [2.0, 3.0]], [[-np.inf, 0.0], [4.0, 5.0]]], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Array with NaN and infinity
    x = tf.constant(np.array([np.nan, np.inf, -np.inf], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with mixed types (should be fine as tf.constant converts)
    x = tf.constant(np.array([1, np.inf, 3.0, -np.inf], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: All zeros
    x = tf.constant(np.zeros((2,2), dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All ones
    x = tf.constant(np.ones((2,2), dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.isinf"] = tf_experimental_numpy_isinf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.isinf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isinf'.")

check_valid('tf.experimental.numpy.isinf', generated_inputs['tf.experimental.numpy.isinf'], lib="tf", suffix=0)
