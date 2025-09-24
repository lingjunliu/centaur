
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_bitcast_inputs():
    list_of_inputs = []

    # Input 1: float32 to int32
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_type = tf.int32
    input_dict = {"input": input_tensor, "type": input_type, "name": "bitcast_float_to_int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32 to float32
    input_tensor = np.array([1, 2, 3, 4], dtype=np.int32)
    input_type = tf.float32
    input_dict = {"input": input_tensor, "type": input_type, "name": "bitcast_int_to_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 to int64
    input_tensor = np.array([1.0, 2.0], dtype=np.float64)
    input_type = tf.int64
    input_dict = {"input": input_tensor, "type": input_type, "name": "bitcast_double_to_long"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64 to float32
    input_tensor = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)
    input_type = tf.float32
    input_dict = {"input": input_tensor, "type": input_type, "name": "bitcast_complex_to_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint8 to int8
    input_tensor = np.array([255, 128, 0], dtype=np.uint8)
    input_type = tf.int8
    input_dict = {"input": input_tensor, "type": input_type, "name": "bitcast_uint8_to_int8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int16 to uint16
    input_tensor = np.array([-1, 0, 1], dtype=np.int16)
    input_type = tf.uint16
    input_dict = {"input": input_tensor, "type": input_type, "name": "bitcast_int16_to_uint16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 to uint8 (shape change)
    input_tensor = np.array([1.0, 2.0], dtype=np.float32)
    input_type = tf.uint8
    input_dict = {"input": input_tensor, "type": input_type, "name": "bitcast_float32_to_uint8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint32 to float32 - REMOVED as uint32 might not be universally supported
    # input_tensor = np.array([1, 2, 3, 4], dtype=np.uint32)
    # input_type = tf.float32
    # input_dict = {"input": input_tensor, "type": input_type, "name": "bitcast_uint32_to_float32"}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  complex128 to float64
    input_tensor = np.array([1 + 1j, 2 + 2j], dtype=np.complex128)
    input_type = tf.float64
    input_dict = {"input": input_tensor, "type": input_type, "name": "bitcast_complex128_to_float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int32 2D array to uint8
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_type = tf.uint8
    input_dict = {"input": input_tensor, "type": input_type, "name": "bitcast_int32_to_uint8_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Bitcast"] = tf_raw_ops_bitcast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Bitcast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Bitcast'.")

check_valid('tf.raw_ops.Bitcast', generated_inputs['tf.raw_ops.Bitcast'], lib="tf", suffix=0)
