
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_identity_inputs():
    list_of_inputs = []

    # Input 1
    n = 0
    dtype = np.int32
    input_dict = {"n": n, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    n = 1
    dtype = np.float32
    input_dict = {"n": n, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    n = 5
    dtype = np.float64
    input_dict = {"n": n, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    n = 2
    dtype = np.int64
    input_dict = {"n": n, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    n = 3
    dtype = np.uint8
    input_dict = {"n": n, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    n = 4
    dtype = np.bool_
    input_dict = {"n": n, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    n = 6
    dtype = np.complex64
    input_dict = {"n": n, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    n = 7
    dtype = np.complex128
    input_dict = {"n": n, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    n = 8
    dtype = np.int16
    input_dict = {"n": n, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    n = 9
    dtype = np.int8
    input_dict = {"n": n, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.identity"] = tf_experimental_numpy_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.identity'.")

check_valid('tf.experimental.numpy.identity', generated_inputs['tf.experimental.numpy.identity'], lib="tf", suffix=0)
