
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_random_randint_inputs():
    list_of_inputs = []

    # Input 1
    low = np.int32(0)
    high = np.int32(10)
    size = np.array([5], dtype=np.int32)
    dtype = np.int32
    input_dict = {"low": low, "high": high, "size": size, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    low = np.int32(-5)
    high = np.int32(5)
    size = np.array([10], dtype=np.int32)
    dtype = np.int64
    input_dict = {"low": low, "high": high, "size": size, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    low = np.int32(1)
    high = np.int32(100)
    size = np.array([2, 3], dtype=np.int32)
    dtype = np.int32
    input_dict = {"low": low, "high": high, "size": size, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    low = np.int32(-100)
    high = np.int32(0)
    size = np.array([4, 2], dtype=np.int32)
    dtype = np.int64
    input_dict = {"low": low, "high": high, "size": size, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    low = np.int32(0)
    high = np.int32(1)
    size = np.array([2, 2, 2], dtype=np.int32)
    dtype = np.int32
    input_dict = {"low": low, "high": high, "size": size, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    low = np.int32(-1)
    high = np.int32(1)
    size = np.array([1, 2, 3, 4], dtype=np.int32)
    dtype = np.int64
    input_dict = {"low": low, "high": high, "size": size, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    low = np.int32(5)
    high = np.int32(10)
    size = np.array([1], dtype=np.int32)
    dtype = np.int32
    input_dict = {"low": low, "high": high, "size": size, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    low = np.int32(-10)
    high = np.int32(-5)
    size = np.array([7], dtype=np.int32)
    dtype = np.int64
    input_dict = {"low": low, "high": high, "size": size, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    low = np.int32(20)
    high = np.int32(21)
    size = np.array([1,1], dtype=np.int32)
    dtype = np.int32
    input_dict = {"low": low, "high": high, "size": size, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    low = np.int32(-21)
    high = np.int32(-20)
    size = np.array([1,1,1], dtype=np.int32)
    dtype = np.int64
    input_dict = {"low": low, "high": high, "size": size, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.random.set_seed(1)
generated_inputs["tf.experimental.numpy.random.randint"] = tf_experimental_numpy_random_randint_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.random.randint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.random.randint'.")

check_valid('tf.experimental.numpy.random.randint', generated_inputs['tf.experimental.numpy.random.randint'], lib="tf", suffix=0)
