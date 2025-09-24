
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_bincount_inputs():
    list_of_inputs = []

    # Input 1
    arr = np.array([1, 1, 2, 3, 2, 4, 4, 5], dtype=np.int32)
    weights = np.array([1, 5, 0, 1, 0, 5, 4, 5], dtype=np.int32)
    minlength = 0
    maxlength = 10
    dtype = tf.int32
    name = "bincount_1"
    axis = None
    binary_output = False
    input_dict = {"arr": arr, "weights": weights, "minlength": minlength, "maxlength": maxlength, "dtype": dtype, "name": name, "axis": axis, "binary_output": binary_output}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    arr = np.array([1, 1, 2, 3, 2, 4, 4, 5], dtype=np.int32)
    weights = None
    minlength = 0
    maxlength = 6
    dtype = tf.int32
    name = "bincount_2"
    input_dict = {"arr": arr, "weights": weights, "minlength": minlength, "maxlength": maxlength, "dtype": dtype, "name": name, "axis": None, "binary_output": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    arr = np.array([[1, 2, 3, 0], [0, 0, 1, 2]], dtype=np.int32)
    weights = None
    minlength = 0
    maxlength = 0
    dtype = tf.int32
    name = "bincount_3"
    input_dict = {"arr": arr, "weights": weights, "minlength": minlength, "maxlength": maxlength, "dtype": dtype, "name": name, "axis": -1, "binary_output": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    arr = np.array([[1, 2, 3, 0], [0, 0, 1, 2]], dtype=np.int32)
    weights = None
    minlength = 0
    maxlength = 0
    dtype = tf.int32
    name = "bincount_4"
    input_dict = {"arr": arr, "weights": weights, "minlength": minlength, "maxlength": maxlength, "dtype": dtype, "name": name, "axis": -1, "binary_output": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    arr = np.array([1, 1, 2, 3, 2, 4, 4, 5], dtype=np.int32)
    weights = np.array([1, 5, 0, 1, 0, 5, 4, 5], dtype=np.int32)
    minlength = 10
    maxlength = 0
    dtype = tf.int32
    name = "bincount_5"
    input_dict = {"arr": arr, "weights": weights, "minlength": minlength, "maxlength": maxlength, "dtype": dtype, "name": name, "axis": None, "binary_output": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    arr = np.array([1, 1, 2, 3, 2, 4, 4, 5], dtype=np.int32)
    weights = np.array([1, 5, 0, 1, 0, 5, 4, 5], dtype=np.int32)
    minlength = 0
    maxlength = 4
    dtype = tf.int32
    name = "bincount_6"
    input_dict = {"arr": arr, "weights": weights, "minlength": minlength, "maxlength": maxlength, "dtype": dtype, "name": name, "axis": None, "binary_output": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    arr = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int32)
    weights = None
    minlength = 7
    maxlength = 0
    dtype = tf.int32
    name = "bincount_7"
    input_dict = {"arr": arr, "weights": weights, "minlength": minlength, "maxlength": maxlength, "dtype": dtype, "name": name, "axis": 0, "binary_output": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    arr = np.array([], dtype=np.int32)
    weights = None
    minlength = 5
    maxlength = 0
    dtype = tf.int32
    name = "bincount_8"
    input_dict = {"arr": arr, "weights": weights, "minlength": minlength, "maxlength": maxlength, "dtype": dtype, "name": name, "axis": None, "binary_output": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    arr = np.array([2, 2, 2, 2, 2], dtype=np.int32)
    weights = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    minlength = 0
    maxlength = 3
    dtype = tf.int32
    name = "bincount_9"
    input_dict = {"arr": arr, "weights": weights, "minlength": minlength, "maxlength": maxlength, "dtype": dtype, "name": name, "axis": None, "binary_output": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    arr = np.array([1, 2, 1, 0, 0, 3], dtype=np.int32)
    weights = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
    minlength = 0
    maxlength = 0
    dtype = tf.float32
    name = "bincount_10"
    input_dict = {"arr": arr, "weights": weights, "minlength": minlength, "maxlength": maxlength, "dtype": dtype, "name": name, "axis": None, "binary_output": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.bincount"] = tf_math_bincount_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.bincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.bincount'.")

check_valid('tf.math.bincount', generated_inputs['tf.math.bincount'], lib="tf", suffix=0)
