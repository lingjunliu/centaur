
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_mod_inputs():
    list_of_inputs = []

    # Input 1: Basic integers
    x = np.array([5, 10, 15], dtype=np.int32)
    y = np.array([2, 3, 4], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "mod_basic_int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcasting
    x = np.array([5, 10, 15], dtype=np.int64)
    y = np.array(2, dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": "mod_broadcast"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative numbers
    x = np.array([-5, -10, 15], dtype=np.int32)
    y = np.array([2, -3, 4], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "mod_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Floats
    x = np.array([5.5, 10.2, 15.7], dtype=np.float32)
    y = np.array([2.1, 3.0, 4.5], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "mod_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Double precision floats
    x = np.array([5.5, 10.2, 15.7], dtype=np.float64)
    y = np.array([2.1, 3.0, 4.5], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "mod_double"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional array
    x = np.array([[5, 10], [15, 20]], dtype=np.int32)
    y = np.array([[2, 3], [4, 5]], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "mod_multi_dim"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large integers
    x = np.array([2**31 - 1, 2**31 - 2], dtype=np.int64)
    y = np.array([3, 5], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": "mod_large_int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different shapes, with broadcasting
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "mod_diff_shape"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Mod"] = tf_raw_ops_mod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Mod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Mod'.")

check_valid('tf.raw_ops.Mod', generated_inputs['tf.raw_ops.Mod'], lib="tf", suffix=0)
