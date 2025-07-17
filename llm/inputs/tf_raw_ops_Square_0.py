
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_square_inputs():
    list_of_inputs = []

    # Input 1: float32 scalar
    x = np.float32(2.0)
    input_dict = {"x": x, "name": "square_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 array
    x = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": "square_float_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32 array
    x = np.array([-2, 0, 3], dtype=np.int32)
    input_dict = {"x": x, "name": "square_int_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 array
    x = np.array([-2.5, 0.5, 3.5], dtype=np.float64)
    input_dict = {"x": x, "name": "square_float64_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int64 array
    x = np.array([-2, 0, 3], dtype=np.int64)
    input_dict = {"x": x, "name": "square_int64_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 array
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "square_2d_float_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64 array
    x = np.array([1+1j, 2-2j], dtype=np.complex64)
    input_dict = {"x": x, "name": "square_complex64_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex128 array
    x = np.array([1+1j, 2-2j], dtype=np.complex128)
    input_dict = {"x": x, "name": "square_complex128_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8 array
    x = np.array([1, 2, 3], dtype=np.uint8)
    input_dict = {"x": x, "name": "square_uint8_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16 array
    x = np.array([-1.0, 2.0], dtype=np.float16).astype(np.float16)
    input_dict = {"x": x, "name": "square_bfloat16_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Square"] = tf_raw_ops_square_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Square' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Square'.")

check_valid('tf.raw_ops.Square', generated_inputs['tf.raw_ops.Square'], lib="tf", suffix=0)
