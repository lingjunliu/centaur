
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_not_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic integers
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([1, 4, 3], dtype=np.int32)
    incompatible_shape_error = True
    name = "not_equal_1"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcasting
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array(2, dtype=np.int32)
    incompatible_shape_error = True
    name = "not_equal_2"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Floats
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.0, 2.5, 3.0], dtype=np.float32)
    incompatible_shape_error = True
    name = "not_equal_3"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Booleans
    x = np.array([True, False, True], dtype=np.bool_)
    y = np.array([False, False, True], dtype=np.bool_)
    incompatible_shape_error = True
    name = "not_equal_4"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[1, 5], [3, 7]], dtype=np.int32)
    incompatible_shape_error = True
    name = "not_equal_5"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different dtypes that can be compared
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([1, 2, 3], dtype=np.int32)
    incompatible_shape_error = True
    name = "not_equal_6"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex numbers
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    y = np.array([1+1j, 2+3j, 3+3j], dtype=np.complex64)
    incompatible_shape_error = True
    name = "not_equal_7"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    y = np.array([[[1, 2], [3, 5]], [[5, 7], [7, 8]]], dtype=np.int32)
    incompatible_shape_error = True
    name = "not_equal_8"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Incompatible shapes with incompatible_shape_error=False
    x = np.array([1, 2], dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    incompatible_shape_error = False
    name = "not_equal_9"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large numbers
    x = np.array([2**31 - 1, 2**30], dtype=np.int64)
    y = np.array([2**31 - 2, 2**30], dtype=np.int64)
    incompatible_shape_error = True
    name = "not_equal_10"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Negative numbers
    x = np.array([-1, -2, 3], dtype=np.int32)
    y = np.array([-1, -4, 3], dtype=np.int32)
    incompatible_shape_error = True
    name = "not_equal_11"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Empty Arrays
    x = np.array([], dtype=np.int32)
    y = np.array([], dtype=np.int32)
    incompatible_shape_error = True
    name = "not_equal_12"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Scalars
    x = np.array(5, dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    incompatible_shape_error = True
    name = "not_equal_13"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14: Strings
    x = np.array("hello", dtype=np.string_)
    y = np.array("world", dtype=np.string_)
    incompatible_shape_error = True
    name = "not_equal_14"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.NotEqual"] = tf_raw_ops_not_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NotEqual' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NotEqual'.")

check_valid('tf.raw_ops.NotEqual', generated_inputs['tf.raw_ops.NotEqual'], lib="tf", suffix=0)
