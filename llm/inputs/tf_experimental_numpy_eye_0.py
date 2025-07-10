
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_eye_inputs():
    list_of_inputs = []

    # Input 1: Basic square identity matrix
    input_dict = {
        "N": 3,
        "M": 3,
        "k": 0,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Rectangular matrix with offset diagonal
    input_dict = {
        "N": 4,
        "M": 5,
        "k": 1,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rectangular matrix with offset diagonal
    input_dict = {
        "N": 5,
        "M": 4,
        "k": -1,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: N only, defaults used for others
    input_dict = {
        "N": 2,
        "M": None,
        "k": 0,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large matrix
    input_dict = {
        "N": 100,
        "M": 100,
        "k": 0,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: N=0
    input_dict = {
        "N": 0,
        "M": 5,
        "k": 0,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: M=0
    input_dict = {
        "N": 5,
        "M": 0,
        "k": 0,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative offset
    input_dict = {
        "N": 5,
        "M": 5,
        "k": -2,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive offset
    input_dict = {
        "N": 5,
        "M": 5,
        "k": 2,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10: int8 datatype
    input_dict = {
        "N": 3,
        "M": 4,
        "k": 0,
        "dtype": np.int8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.eye"] = tf_experimental_numpy_eye_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.eye' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.eye'.")

check_valid('tf.experimental.numpy.eye', generated_inputs['tf.experimental.numpy.eye'], lib="tf", suffix=0)
