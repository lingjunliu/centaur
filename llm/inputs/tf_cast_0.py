
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_cast_inputs():
    list_of_inputs = []

    # Input 1: Basic int32 to float32
    x = np.array([1, 2, 3], dtype=np.int32)
    dtype = tf.float32
    name = "int_to_float"
    input_dict = {"x": x, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 to int64
    x = np.array([1.5, 2.7, 3.9], dtype=np.float64)
    dtype = tf.int64
    name = "float_to_int"
    input_dict = {"x": x, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int8 to float16
    x = np.array([-1, 0, 1], dtype=np.int8)
    dtype = tf.float16
    name = "int_to_float16"
    input_dict = {"x": x, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64 to float32
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    dtype = tf.float32
    name = "complex_to_float"
    input_dict = {"x": x, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32 to complex128
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dtype = tf.complex128
    name = "float_to_complex"
    input_dict = {"x": x, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int32 to int8 (with clipping)
    x = np.array([100, -50, 200], dtype=np.int32)
    dtype = tf.int8
    name = "int_to_smaller_int"
    input_dict = {"x": x, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multi-dimensional array (float32 to int32)
    x = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
    dtype = tf.int32
    name = "2d_float_to_int"
    input_dict = {"x": x, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: bfloat16 to float32 - REMOVED
    #x = np.array([1.0, 2.0, 3.0], dtype=tf.bfloat16.as_numpy_dtype)
    #dtype = tf.float32
    #name = "bfloat16_to_float32"
    #input_dict = {"x": x, "dtype": dtype, "name": name}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int64 to float64
    x = np.array([1, 2, 3], dtype=np.int64)
    dtype = tf.float64
    name = "int64_to_float64"
    input_dict = {"x": x, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: int16 to float64
    x = np.array([1, -2, 3], dtype=np.int16)
    dtype = tf.float64
    name = "int16_to_float64"
    input_dict = {"x": x, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: float16 to float32
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    dtype = tf.float32
    name = "float16_to_float32"
    input_dict = {"x": x, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14: float32 to float64
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dtype = tf.float64
    name = "float32_to_float64"
    input_dict = {"x": x, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.cast"] = tf_cast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.cast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.cast'.")

check_valid('tf.cast', generated_inputs['tf.cast'], lib="tf", suffix=0)
