
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ones_like_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, 2D
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"x": x, "name": "my_ones"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bool, 3D
    x = np.array([[[True, False], [False, True]], [[True, True], [False, False]]], dtype=np.bool_)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64, 1D
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {"x": x, "name": "complex_ones"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int64, 2D with negative values
    x = np.array([[-1, 2], [-3, 4]], dtype=np.int64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8, 3D
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64, empty array
    x = np.array([], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16, 2D
    x = np.array([[1, 2], [3, 4]], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint32, 1D
    x = np.array([1, 2, 3], dtype=np.uint32)
    input_dict = {"x": x, "name": "uint32_ones"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128, 2D
    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: int8, 2D with different range
    x = np.array([[-128, 127], [0, 1]], dtype=np.int8)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: uint16, 4D
    x = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.uint16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.OnesLike"] = tf_raw_ops_ones_like_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.OnesLike' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OnesLike'.")

check_valid('tf.raw_ops.OnesLike', generated_inputs['tf.raw_ops.OnesLike'], lib="tf", suffix=0)
