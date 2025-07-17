
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integers
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    y = np.array([1, 2, 0, 4, 0], dtype=np.int32)
    incompatible_shape_error = True
    name = "equal_op_1"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcasting scalar
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array(2, dtype=np.int32)
    incompatible_shape_error = True
    name = "equal_op_2"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Floating point numbers
    x = np.array([1.0, 2.5, 3.7], dtype=np.float32)
    y = np.array([1.0, 2.5, 3.6], dtype=np.float32)
    incompatible_shape_error = True
    name = "equal_op_3"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean values
    x = np.array([True, False, True], dtype=np.bool_)
    y = np.array([True, True, False], dtype=np.bool_)
    incompatible_shape_error = True
    name = "equal_op_4"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[1, 2], [0, 4]], dtype=np.int32)
    incompatible_shape_error = True
    name = "equal_op_5"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting with 2D arrays
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([1, 2], dtype=np.int32)
    incompatible_shape_error = True
    name = "equal_op_6"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger integers
    x = np.array([1000000000, 2000000000], dtype=np.int64)
    y = np.array([1000000000, 2000000001], dtype=np.int64)
    incompatible_shape_error = True
    name = "equal_op_7"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values
    x = np.array([-1, -2, 3], dtype=np.int32)
    y = np.array([-1, 2, -3], dtype=np.int32)
    incompatible_shape_error = True
    name = "equal_op_8"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D arrays
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    y = np.array([[[1, 2], [3, 0]], [[5, 6], [0, 8]]], dtype=np.int32)
    incompatible_shape_error = True
    name = "equal_op_9"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different shape that can be broadcast
    x = np.array([[1, 2, 3]], dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    incompatible_shape_error = True
    name = "equal_op_10"
    input_dict = {"x": x, "y": y, "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Equal"] = tf_raw_ops_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Equal'.")

check_valid('tf.raw_ops.Equal', generated_inputs['tf.raw_ops.Equal'], lib="tf", suffix=0)
