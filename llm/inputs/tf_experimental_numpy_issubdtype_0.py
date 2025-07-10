
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_issubdtype_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "arg1": np.int32,
        "arg2": np.integer
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "arg1": np.float64,
        "arg2": np.floating
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "arg1": np.complex128,
        "arg2": np.complexfloating
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "arg1": np.bool_,
        "arg2": np.generic
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "arg1": np.int8,
        "arg2": np.number
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "arg1": np.uint64,
        "arg2": np.unsignedinteger
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "arg1": np.float16,
        "arg2": np.inexact
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "arg1": np.str_,
        "arg2": np.flexible
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "arg1": np.datetime64,
        "arg2": np.datetime64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "arg1": np.timedelta64,
        "arg2": np.timedelta64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_dict = {
        "arg1": np.int16,
        "arg2": np.signedinteger
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.issubdtype"] = tf_experimental_numpy_issubdtype_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.issubdtype' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.issubdtype'.")

check_valid('tf.experimental.numpy.issubdtype', generated_inputs['tf.experimental.numpy.issubdtype'], lib="tf", suffix=0)
