
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_add_inputs():
    list_of_inputs = []

    # Input 1: Basic addition of two scalars
    x = np.array(5, dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "add_scalars"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Addition of two 1D arrays
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    y = np.array([6, 7, 8, 9, 10], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "add_1d_arrays"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Addition of two 2D arrays
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[5, 6], [7, 8]], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": "add_2d_arrays"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Addition with negative values
    x = np.array([-1, -2, -3], dtype=np.int8)
    y = np.array([4, 5, 6], dtype=np.int8)
    input_dict = {"x": x, "y": y, "name": "add_negative_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Addition with zero
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([0, 0, 0], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": "add_zero"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Addition with complex numbers
    x = np.array([1+1j, 2+2j], dtype=np.complex64)
    y = np.array([3+3j, 4+4j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": "add_complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Addition with bfloat16
    x = np.array([1, 2, 3], dtype=np.float32).astype(np.float16)
    y = np.array([4, 5, 6], dtype=np.float32).astype(np.float16)
    input_dict = {"x": x, "y": y, "name": "add_bfloat16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Addition with half
    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": "add_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Addition with broadcasting - scalar and array
    x = np.array(2, dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "add_broadcasting"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Addition with different shapes that can be broadcast
    x = np.array([[1, 2, 3]], dtype=np.float32)
    y = np.array([[4], [5]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "add_broadcast_shapes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Add"] = tf_raw_ops_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Add'.")

check_valid('tf.raw_ops.Add', generated_inputs['tf.raw_ops.Add'], lib="tf", suffix=0)
