
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_isinf_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive and negative inf
    x = np.array([np.inf, -np.inf, 1.0, 0.0, -1.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: All inf values (positive and negative)
    x = np.array([np.inf, -np.inf, np.inf, -np.inf], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": "all_inf"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: No inf values
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed values including nan and inf
    x = np.array([np.inf, -np.inf, np.nan, 1.0, 0.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": "mixed"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional array
    x = np.array([[np.inf, 1.0], [-np.inf, 2.0]], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64 array
    x = np.array([np.inf, -np.inf, 1.0, 0.0], dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: An empty array
    x = np.array([], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": "empty"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with zeros
    x = np.array([0.0, 0.0, np.inf, -np.inf], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": "zeros"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multi-dimensional array with float64
    x = np.array([[np.inf, 1.0], [-np.inf, 2.0]], dtype=np.float64)
    input_dict = {"x": tf.constant(x, dtype=tf.float64), "name": 'float64_multi'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with many dimensions and some inf values
    x = np.zeros((2, 3, 4), dtype=np.float32)
    x[0, 0, 0] = np.inf
    x[1, 2, 3] = -np.inf
    input_dict = {"x": tf.constant(x), "name": "many_dims"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.IsInf"] = tf_raw_ops_isinf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.IsInf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.IsInf'.")

check_valid('tf.raw_ops.IsInf', generated_inputs['tf.raw_ops.IsInf'], lib="tf", suffix=0)
