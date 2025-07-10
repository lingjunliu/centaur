
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_finfo_inputs():
    list_of_inputs = []

    # Input 1: float16
    dtype = np.float16
    input_dict = {"dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32
    dtype = np.float32
    input_dict = {"dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64
    dtype = np.float64
    input_dict = {"dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64
    dtype = np.complex64
    input_dict = {"dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex128
    dtype = np.complex128
    input_dict = {"dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: np.dtype('float16')
    dtype = np.dtype('float16')
    input_dict = {"dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: np.dtype('float32')
    dtype = np.dtype('float32')
    input_dict = {"dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: np.dtype('float64')
    dtype = np.dtype('float64')
    input_dict = {"dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: np.dtype('complex64')
    dtype = np.dtype('complex64')
    input_dict = {"dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: np.dtype('complex128')
    dtype = np.dtype('complex128')
    input_dict = {"dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.finfo"] = tf_experimental_numpy_finfo_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.finfo' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.finfo'.")

check_valid('tf.experimental.numpy.finfo', generated_inputs['tf.experimental.numpy.finfo'], lib="tf", suffix=0)
