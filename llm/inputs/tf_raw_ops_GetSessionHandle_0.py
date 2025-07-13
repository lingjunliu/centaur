
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_get_session_handle_inputs():
    list_of_inputs = []

    # Input 1: Simple float tensor
    value = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    name = "float_tensor"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int tensor with a name
    value = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    name = "int_tensor"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Boolean tensor
    value = tf.constant(np.array([True, False, True], dtype=np.bool_))
    name = "bool_tensor"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D tensor
    value = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    name = "2d_tensor"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor
    value = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32))
    name = "3d_tensor"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty string as name
    value = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    name = ""
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large int tensor
    value = tf.constant(np.array([1000, 2000, 3000], dtype=np.int32))
    name = "large_int_tensor"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with negative values
    value = tf.constant(np.array([-1, -2, -3], dtype=np.int32))
    name = "negative_tensor"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: String tensor.
    value = tf.constant(np.array(["hello"], dtype=np.string_))
    name = "string_tensor"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with a long name.
    value = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    name = "a_very_long_name_for_the_tensor"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.GetSessionHandle"] = tf_raw_ops_get_session_handle_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.GetSessionHandle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GetSessionHandle'.")

check_valid('tf.raw_ops.GetSessionHandle', generated_inputs['tf.raw_ops.GetSessionHandle'], lib="tf", suffix=0)
