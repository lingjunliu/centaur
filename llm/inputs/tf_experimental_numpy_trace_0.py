
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_trace_inputs():
    list_of_inputs = []

    # Input 1
    a = tf.constant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    offset = 0
    axis1 = 0
    axis2 = 1
    dtype = tf.int32

    input_dict = {
        "a": a,
        "offset": offset,
        "axis1": axis1,
        "axis2": axis2,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    offset = 1
    axis1 = 0
    axis2 = 1
    dtype = tf.float32

    input_dict = {
        "a": a,
        "offset": offset,
        "axis1": axis1,
        "axis2": axis2,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]))
    offset = -1
    axis1 = 0
    axis2 = 1
    dtype = tf.int64

    input_dict = {
        "a": a,
        "offset": offset,
        "axis1": axis1,
        "axis2": axis2,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    offset = 0
    axis1 = 1
    axis2 = 2
    dtype = tf.float64

    input_dict = {
        "a": a,
        "offset": offset,
        "axis1": axis1,
        "axis2": axis2,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = tf.constant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    offset = 2
    axis1 = 0
    axis2 = 1
    dtype = tf.float16

    input_dict = {
        "a": a,
        "offset": offset,
        "axis1": axis1,
        "axis2": axis2,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    offset = -1
    axis1 = 0
    axis2 = 1
    dtype = tf.complex64

    input_dict = {
        "a": a,
        "offset": offset,
        "axis1": axis1,
        "axis2": axis2,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = tf.constant(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    offset = 0
    axis1 = 0
    axis2 = 2
    dtype = tf.complex128

    input_dict = {
        "a": a,
        "offset": offset,
        "axis1": axis1,
        "axis2": axis2,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = tf.constant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    offset = -2
    axis1 = 0
    axis2 = 1
    dtype = tf.int32

    input_dict = {
        "a": a,
        "offset": offset,
        "axis1": axis1,
        "axis2": axis2,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    offset = 0
    axis1 = 0
    axis2 = 1
    dtype = tf.int8

    input_dict = {
        "a": a,
        "offset": offset,
        "axis1": axis1,
        "axis2": axis2,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    a = tf.constant(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    offset = 1
    axis1 = 0
    axis2 = 1
    dtype = tf.int16

    input_dict = {
        "a": a,
        "offset": offset,
        "axis1": axis1,
        "axis2": axis2,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs = {}
generated_inputs["tf.experimental.numpy.trace"] = tf_experimental_numpy_trace_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.trace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.trace'.")

check_valid('tf.experimental.numpy.trace', generated_inputs['tf.experimental.numpy.trace'], lib="tf", suffix=0)
