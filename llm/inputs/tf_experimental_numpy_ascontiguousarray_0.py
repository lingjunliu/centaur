
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_ascontiguousarray_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    a = tf.constant(np.array([1, 2, 3]))
    dtype = None
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array with a specific dtype
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    dtype = np.int64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Array with negative values
    a = tf.constant(np.array([-1, -2, 3]))
    dtype = None
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Array with float values
    a = tf.constant(np.array([1.5, 2.5, 3.5]))
    dtype = np.float64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Array with zero values
    a = tf.constant(np.array([0, 0, 0]))
    dtype = None
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty array
    a = tf.constant(np.array([]))
    dtype = None
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Array with different data types
    a = tf.constant(np.array([1, 2, 3], dtype=np.int16))
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: A larger multi-dimensional array
    a = tf.constant(np.random.rand(2, 3, 4).astype(np.float32))
    dtype = None
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs = {}
generated_inputs["tf.experimental.numpy.ascontiguousarray"] = tf_experimental_numpy_ascontiguousarray_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.ascontiguousarray' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.ascontiguousarray'.")

check_valid('tf.experimental.numpy.ascontiguousarray', generated_inputs['tf.experimental.numpy.ascontiguousarray'], lib="tf", suffix=0)
