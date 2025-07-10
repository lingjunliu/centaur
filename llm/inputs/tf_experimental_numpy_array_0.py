
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_array_inputs():
    list_of_inputs = []

    # Input 1: Basic array creation
    val = tf.constant([1, 2, 3], dtype=tf.int32).numpy()
    dtype = tf.int32
    copy_flag = True
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Change dtype
    val = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    dtype = tf.int64
    copy_flag = False
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: ndmin > 0
    val = tf.constant([1, 2, 3], dtype=tf.int32).numpy()
    dtype = tf.int32
    copy_flag = True
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array
    val = tf.constant([[1, 2], [3, 4]], dtype=tf.int32).numpy()
    dtype = tf.int32
    copy_flag = False
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different dtype for val
    val = tf.constant([1.5, 2.5, 3.5], dtype=tf.float64).numpy()
    dtype = tf.float64
    copy_flag = True
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex numbers
    val = tf.constant([1+1j, 2+2j, 3+3j], dtype=tf.complex128).numpy()
    dtype = tf.complex128
    copy_flag = False
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean values
    val = tf.constant([True, False, True], dtype=tf.bool).numpy()
    dtype = tf.bool
    copy_flag = True
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array
    val = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.int32).numpy()
    dtype = tf.int32
    copy_flag = False
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative numbers
    val = tf.constant([-1, -2, -3], dtype=tf.int32).numpy()
    dtype = tf.int32
    copy_flag = True
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: ndmin greater than the tensor rank
    val = tf.constant([1, 2], dtype=tf.int32).numpy()
    dtype = tf.int32
    copy_flag = False
    ndmin = 3
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.array"] = tf_experimental_numpy_array_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.array' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.array'.")

check_valid('tf.experimental.numpy.array', generated_inputs['tf.experimental.numpy.array'], lib="tf", suffix=0)
