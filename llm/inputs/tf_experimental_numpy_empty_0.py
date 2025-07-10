
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_empty_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float64
    shape = (2, 3)
    dtype = np.float64
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer type
    shape = (5,)
    dtype = np.int32
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional array
    shape = (2, 2, 2)
    dtype = np.float32
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero-sized array
    shape = (0,)
    dtype = np.int64
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different integer type
    shape = (3, 4)
    dtype = np.uint8
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex number type
    shape = (2,)
    dtype = np.complex64
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean type
    shape = (4, 1)
    dtype = np.bool_
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another shape and float type
    shape = (1, 5, 1)
    dtype = np.float16
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large shape
    shape = (100,)
    dtype = np.float32
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Shape with a single element
    shape = (1,)
    dtype = np.int8
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.empty"] = tf_experimental_numpy_empty_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.empty' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.empty'.")

check_valid('tf.experimental.numpy.empty', generated_inputs['tf.experimental.numpy.empty'], lib="tf", suffix=0)
