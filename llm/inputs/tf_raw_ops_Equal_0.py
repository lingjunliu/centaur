
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic equal with boolean tensors
    x = np.array([True, False, True], dtype=np.bool_)
    y = np.array([True, True, False], dtype=np.bool_)
    incompatible_shape_error = True
    name = "equal_bool"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Equal with integer tensors
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([1, 2, 4], dtype=np.int32)
    incompatible_shape_error = False
    name = "equal_int"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Equal with float tensors
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.0, 2.1, 3.0], dtype=np.float32)
    incompatible_shape_error = True
    name = "equal_float"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Equal with string tensors
    x = np.array(["a", "b", "c"], dtype=np.str_)
    y = np.array(["a", "b", "d"], dtype=np.str_)
    incompatible_shape_error = False
    name = "equal_string"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Equal with 2D tensors
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[1, 2], [3, 5]], dtype=np.int32)
    incompatible_shape_error = True
    name = "equal_2d"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Equal with different shapes (broadcasting)
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array(2, dtype=np.int32)
    incompatible_shape_error = True
    name = "equal_broadcast"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Equal with 3D tensors
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    y = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 9]]], dtype=np.int32)
    incompatible_shape_error = True
    name = "equal_3d"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Equal with negative numbers
    x = np.array([-1, -2, 3], dtype=np.int32)
    y = np.array([-1, 2, -3], dtype=np.int32)
    incompatible_shape_error = False
    name = "equal_negative"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Equal with empty arrays
    x = np.array([], dtype=np.int32)
    y = np.array([], dtype=np.int32)
    incompatible_shape_error = True
    name = "equal_empty"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "incompatible_shape_error": incompatible_shape_error, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Equal with complex numbers
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    y = np.array([1+1j, 2+3j, 3+3j], dtype=np.complex128)
    incompatible_shape_error = False
    name = "equal_complex"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "incompatible_shape_error": incompatible_shape_error, "name": name}
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
