
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_as_string_inputs():
    list_of_inputs = []

    # Input 1: Basic integer array
    input_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {"input": input_array, "precision": -1, "scientific": False, "shortest": False, "width": -1, "fill": "", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float array with precision
    input_array = np.array([3.14159, 2.71828, 1.61803], dtype=np.float32)
    input_dict = {"input": input_array, "precision": 2, "scientific": False, "shortest": False, "width": -1, "fill": "", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Integer array with width and fill
    input_array = np.array([1, 10, 100], dtype=np.int32)
    input_dict = {"input": input_array, "precision": -1, "scientific": False, "shortest": False, "width": 4, "fill": "0", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float array with scientific notation
    input_array = np.array([0.0001, 10000.0], dtype=np.float64)
    input_dict = {"input": input_array, "precision": 3, "scientific": True, "shortest": False, "width": -1, "fill": "", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boolean array
    input_array = np.array([True, False, True, True, False], dtype=np.bool_)
    input_dict = {"input": input_array, "precision": -1, "scientific": False, "shortest": False, "width": -1, "fill": "", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float array
    input_array = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
    input_dict = {"input": input_array, "precision": 1, "scientific": False, "shortest": False, "width": -1, "fill": "", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Integer with width and fill (space)
    input_array = np.array([7, 123, 4], dtype=np.int32)
    input_dict = {"input": input_array, "precision": -1, "scientific": False, "shortest": False, "width": 5, "fill": "", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Shortest representation for floats
    input_array = np.array([0.001, 1000.0, 1.0], dtype=np.float32)
    input_dict = {"input": input_array, "precision": 3, "scientific": False, "shortest": True, "width": -1, "fill": "", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex numbers
    input_array = np.array([1+1j, 2-2j, 3+0j], dtype=np.complex64)
    input_dict = {"input": input_array, "precision": 2, "scientific": False, "shortest": False, "width": -1, "fill": "", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint8 array
    input_array = np.array([255, 128, 0], dtype=np.uint8)
    input_dict = {"input": input_array, "precision": -1, "scientific": False, "shortest": False, "width": -1, "fill": "", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.as_string"] = tf_strings_as_string_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.as_string' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.as_string'.")

check_valid('tf.strings.as_string', generated_inputs['tf.strings.as_string'], lib="tf", suffix=0)
