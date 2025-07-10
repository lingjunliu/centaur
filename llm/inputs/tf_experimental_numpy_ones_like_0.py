
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_ones_like_inputs():
    list_of_inputs = []
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    # Input 1: int32, 1D
    a = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    dtype = tf.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 2D
    a = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    dtype = tf.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int64, 3D
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64))
    dtype = tf.int64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: bool, 1D
    a = tf.constant(np.array([True, False, True], dtype=np.bool_))
    dtype = tf.bool
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64, 2D with different dtype
    a = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    dtype = tf.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8, 2D
    a = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.uint8))
    dtype = tf.uint8
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int16, 1D
    a = tf.constant(np.array([1, 2, 3, -4], dtype=np.int16))
    dtype = tf.int16
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: float16, 2D
    a = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16))
    dtype = tf.float16
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int8, 1D
    a = tf.constant(np.array([1, 2, 3, -4], dtype=np.int8))
    dtype = tf.int8
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex64, 1D - Removing complex as it causes errors with numpy behavior
    # a = tf.constant(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64))
    # dtype = tf.complex64
    # input_dict = {"a": a, "dtype": dtype}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.ones_like"] = tf_experimental_numpy_ones_like_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.ones_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.ones_like'.")

check_valid('tf.experimental.numpy.ones_like', generated_inputs['tf.experimental.numpy.ones_like'], lib="tf", suffix=0)
