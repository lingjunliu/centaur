
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_get_session_handle_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor of integers
    value = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {"value": tf.convert_to_tensor(value), "name": "handle_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor of floats
    value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"value": tf.convert_to_tensor(value), "name": "handle_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tensor of strings
    value = np.array(["hello", "world"], dtype=np.string_)
    input_dict = {"value": tf.convert_to_tensor(value), "name": "handle_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor of booleans
    value = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"value": tf.convert_to_tensor(value), "name": "handle_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with negative values
    value = np.array([-1, -2, 3, -4, 5], dtype=np.int32)
    input_dict = {"value": tf.convert_to_tensor(value), "name": "handle_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.GetSessionHandle"] = tf_raw_ops_get_session_handle_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.GetSessionHandle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GetSessionHandle'.")

check_valid('tf.raw_ops.GetSessionHandle', generated_inputs['tf.raw_ops.GetSessionHandle'], lib="tf", suffix=0)
