
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_result_type_inputs():
    list_of_inputs = []

    # Input 1: Basic integers
    input_dict = {"arrays_and_dtypes": [np.int32, np.int64]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic floats
    input_dict = {"arrays_and_dtypes": [np.float32, np.float64]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixed integers and floats
    input_dict = {"arrays_and_dtypes": [np.int32, np.float64]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex numbers
    input_dict = {"arrays_and_dtypes": [np.complex64, np.complex128]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Booleans and integers
    input_dict = {"arrays_and_dtypes": [np.bool_, np.int8]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Array and a dtype
    a = np.array([1, 2, 3])
    input_dict = {"arrays_and_dtypes": [a, np.float64]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multiple arrays
    a = np.array([1, 2, 3])
    b = np.array([1.0, 2.0, 3.0])
    input_dict = {"arrays_and_dtypes": [a, b, np.int64]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different array shapes
    a = np.array([[1, 2], [3, 4]])
    b = np.array([1.0, 2.0])
    input_dict = {"arrays_and_dtypes": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Unsigned integers
    input_dict = {"arrays_and_dtypes": [np.uint8, np.uint32]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.result_type"] = tf_experimental_numpy_result_type_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.result_type' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.result_type'.")

check_valid('tf.experimental.numpy.result_type', generated_inputs['tf.experimental.numpy.result_type'], lib="tf", suffix=0)
