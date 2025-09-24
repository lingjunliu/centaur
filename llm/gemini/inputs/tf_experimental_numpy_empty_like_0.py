
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_empty_like_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array, default dtype
    a = np.array([1, 2, 3])
    dtype = None
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float32 dtype
    a = np.array([[1, 2], [3, 4]])
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, int64 dtype
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dtype = np.int64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Array with negative values, complex64 dtype
    a = np.array([-1, -2, 3])
    dtype = np.complex64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty array, bool dtype
    a = np.array([])
    dtype = np.bool_
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Multi-dimensional array with mixed data types (will be cast), int8
    a = np.array([[1.5, 2.3], [3.1, 4.9]])
    dtype = np.int8
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large array, uint16
    a = np.random.rand(100, 100).astype(np.float32)
    dtype = np.uint16
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar value, int32
    a = np.array(5)
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Rank 4 tensor, float64
    a = np.random.rand(2, 3, 4, 5).astype(np.float32)
    dtype = np.float64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boolean array, float16
    a = np.array([True, False, True])
    dtype = np.float16
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.empty_like"] = tf_experimental_numpy_empty_like_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.empty_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.empty_like'.")

check_valid('tf.experimental.numpy.empty_like', generated_inputs['tf.experimental.numpy.empty_like'], lib="tf", suffix=0)
