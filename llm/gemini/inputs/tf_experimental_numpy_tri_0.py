
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_tri_inputs():
    list_of_inputs = []

    # Input 1
    N = 3
    M = 5
    k = 0
    dtype = np.int32
    input_dict = {"N": N, "M": M, "k": k, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    N = 4
    M = 4
    k = 1
    dtype = np.float32
    input_dict = {"N": N, "M": M, "k": k, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    N = 5
    M = 6
    k = -1
    dtype = np.bool_
    input_dict = {"N": N, "M": M, "k": k, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    N = 2
    M = 7
    k = 2
    dtype = np.float64
    input_dict = {"N": N, "M": M, "k": k, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    N = 6
    M = 2
    k = -2
    dtype = np.int64
    input_dict = {"N": N, "M": M, "k": k, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    N = 7
    M = None
    k = 0
    dtype = np.int32
    input_dict = {"N": N, "M": M, "k": k, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    N = 8
    M = None
    k = 1
    dtype = np.float32
    input_dict = {"N": N, "M": M, "k": k, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    N = 9
    M = None
    k = -1
    dtype = np.bool_
    input_dict = {"N": N, "M": M, "k": k, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    N = 1
    M = None
    k = 2
    dtype = np.float64
    input_dict = {"N": N, "M": M, "k": k, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    N = 10
    M = None
    k = -2
    dtype = np.int64
    input_dict = {"N": N, "M": M, "k": k, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    N = 3
    M = 3
    k = 3
    dtype = np.int8
    input_dict = {"N": N, "M": M, "k": k, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    N = 3
    M = 3
    k = -3
    dtype = np.int8
    input_dict = {"N": N, "M": M, "k": k, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.tri"] = tf_experimental_numpy_tri_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.tri' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.tri'.")

check_valid('tf.experimental.numpy.tri', generated_inputs['tf.experimental.numpy.tri'], lib="tf", suffix=0)
