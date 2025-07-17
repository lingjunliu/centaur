
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_get_session_handle_v2_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    value = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"value": tf.constant(value), "name": "basic_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with float values
    value = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    input_dict = {"value": tf.constant(value), "name": "2d_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor with boolean values
    value = np.array([[[True, False], [False, True]], [[True, True], [False, False]]], dtype=np.bool_)
    input_dict = {"value": tf.constant(value.astype(np.int32)), "name": "3d_bool"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty tensor
    value = np.array([], dtype=np.int32)
    input_dict = {"value": tf.constant(value.reshape(0)), "name": "empty"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with negative values
    value = np.array([-1, -2, -3], dtype=np.int32)
    input_dict = {"value": tf.constant(value), "name": "negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large tensor
    value = np.random.rand(10, 10).astype(np.float32)
    input_dict = {"value": tf.constant(value), "name": "large"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with zeros
    value = np.zeros((5, 5), dtype=np.float32)
    input_dict = {"value": tf.constant(value), "name": "zeros"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with ones
    value = np.ones((3, 3, 3), dtype=np.int32)
    input_dict = {"value": tf.constant(value), "name": "ones"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with different data types (float64)
    value = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"value": tf.constant(value.astype(np.float32)), "name": "float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D tensor
    value = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict = {"value": tf.constant(value), "name": "4d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.GetSessionHandleV2"] = tf_raw_ops_get_session_handle_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.GetSessionHandleV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GetSessionHandleV2'.")

check_valid('tf.raw_ops.GetSessionHandleV2', generated_inputs['tf.raw_ops.GetSessionHandleV2'], lib="tf", suffix=0)
