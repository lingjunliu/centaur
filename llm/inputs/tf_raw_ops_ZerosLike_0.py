
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ZerosLike_inputs():
    list_of_inputs = []

    # Input 1: Integer Tensor
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict = {"x": x, "name": "zeros_like_int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float Tensor
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "zeros_like_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex Tensor
    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    input_dict = {"x": x, "name": "zeros_like_complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Bool Tensor
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    input_dict = {"x": x, "name": "zeros_like_bool"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional Tensor
    x = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict = {"x": x, "name": "zeros_like_multi"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with negative values
    x = np.array([[-1, -2], [-3, -4]], dtype=np.int64)
    input_dict = {"x": x, "name": "zeros_like_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero dimensional tensor (scalar)
    x = np.array(5, dtype=np.int32)
    input_dict = {"x": x, "name": "zeros_like_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with all zeros
    x = np.zeros((2, 2), dtype=np.float32)
    input_dict = {"x": x, "name": "zeros_like_zeros"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with different data types
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    input_dict = {"x": x, "name": "zeros_like_uint8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger tensor
    x = np.random.randint(0, 100, size=(10, 10), dtype=np.int32)
    input_dict = {"x": x, "name": "zeros_like_large"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ZerosLike"] = tf_raw_ops_ZerosLike_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ZerosLike' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ZerosLike'.")

check_valid('tf.raw_ops.ZerosLike', generated_inputs['tf.raw_ops.ZerosLike'], lib="tf", suffix=0)
