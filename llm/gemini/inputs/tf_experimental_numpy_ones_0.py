
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_ones_inputs():
    list_of_inputs = []

    # Input 1
    shape = 5
    dtype = np.float32
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = (2, 3)
    dtype = np.int32
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = (2, 2, 2)
    dtype = np.float64
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = (1, 5, 1, 3)
    dtype = np.int8
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = 0
    dtype = np.bool_
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = (0, 5)
    dtype = np.uint16
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    shape = (5, 0, 2)
    dtype = np.complex64
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = (1,)
    dtype = np.int64
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = (3,1)
    dtype = np.uint8
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = (1,1,1)
    dtype = np.float16
    input_dict = {"shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.ones"] = tf_experimental_numpy_ones_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.ones' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.ones'.")

check_valid('tf.experimental.numpy.ones', generated_inputs['tf.experimental.numpy.ones'], lib="tf", suffix=0)
