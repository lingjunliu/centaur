
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_cumprod_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([1, 2, 3], dtype=np.int32)
    axis = 0
    dtype = tf.int32
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    axis = 1
    dtype = tf.float32
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[-1, 2], [-3, 4]], dtype=np.int64)
    axis = 0
    dtype = tf.int64
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    axis = 0
    dtype = tf.float64
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis = 2
    dtype = tf.int32
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([[1, 2, 3]], dtype=np.float32)
    axis = 1
    dtype = tf.float32
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([1, 2, 3, 4], dtype=np.int16)
    axis = 0
    dtype = tf.int16
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    axis = 1
    dtype = tf.float16
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([1, 0, 3], dtype=np.int32)
    axis = 0
    dtype = tf.int32
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    a = np.array([[1, 2], [3, -4]], dtype=np.int8)
    axis = 0
    dtype = tf.int8
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.cumprod"] = tf_experimental_numpy_cumprod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.cumprod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.cumprod'.")

check_valid('tf.experimental.numpy.cumprod', generated_inputs['tf.experimental.numpy.cumprod'], lib="tf", suffix=0)
