
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_log2_inputs():
    list_of_inputs = []

    # Input 1: Basic positive integers
    x = tf.constant(np.array([1, 2, 4, 8, 16], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive floats
    x = tf.constant(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero and one
    x = tf.constant(np.array([0.0, 1.0], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional array (2D)
    x = tf.constant(np.array([[1, 2, 4], [8, 16, 32]], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.log2"] = tf_experimental_numpy_log2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.log2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.log2'.")

check_valid('tf.experimental.numpy.log2', generated_inputs['tf.experimental.numpy.log2'], lib="tf", suffix=0)
