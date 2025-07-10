
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_bincount_inputs():
    list_of_inputs = []

    # Input 1: Basic example with numpy array
    values = np.array([1, 2, 3, 2, 1], dtype=np.int64)
    weights = None
    axis = 0
    minlength = None
    maxlength = None
    binary_output = False
    name = None
    input_dict = {"values": values, "weights": weights, "axis": axis, "minlength": minlength, "maxlength": maxlength, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With weights
    values = np.array([1, 2, 3, 2, 1], dtype=np.int64)
    weights = np.array([0.5, 1.0, 1.5, 1.0, 0.5], dtype=np.float32)
    axis = 0
    minlength = None
    maxlength = None
    binary_output = False
    name = None
    input_dict = {"values": values, "weights": weights, "axis": axis, "minlength": minlength, "maxlength": maxlength, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With minlength
    values = np.array([1, 2, 3], dtype=np.int64)
    weights = None
    axis = 0
    minlength = 5
    maxlength = None
    binary_output = False
    name = None
    input_dict = {"values": values, "weights": weights, "axis": axis, "minlength": minlength, "maxlength": maxlength, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With maxlength
    values = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    weights = None
    axis = 0
    minlength = None
    maxlength = 4
    binary_output = False
    name = None
    input_dict = {"values": values, "weights": weights, "axis": axis, "minlength": minlength, "maxlength": maxlength, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: With binary_output
    values = np.array([1, 2, 3, 2, 1], dtype=np.int64)
    weights = None
    axis = 0
    minlength = None
    maxlength = None
    binary_output = True
    name = None
    input_dict = {"values": values, "weights": weights, "axis": axis, "minlength": minlength, "maxlength": maxlength, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: With axis=-1 and 2D array
    values = np.array([[1, 2, 3], [2, 1, 0]], dtype=np.int64)
    weights = None
    axis = -1
    minlength = None
    maxlength = None
    binary_output = False
    name = None
    input_dict = {"values": values, "weights": weights, "axis": axis, "minlength": minlength, "maxlength": maxlength, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: With minlength and maxlength
    values = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    weights = None
    axis = 0
    minlength = 3
    maxlength = 4
    binary_output = False
    name = None
    input_dict = {"values": values, "weights": weights, "axis": axis, "minlength": minlength, "maxlength": maxlength, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty array
    values = np.array([], dtype=np.int64)
    weights = None
    axis = 0
    minlength = 5
    maxlength = 10
    binary_output = False
    name = None
    input_dict = {"values": values, "weights": weights, "axis": axis, "minlength": minlength, "maxlength": maxlength, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Weights as array of ones
    values = np.array([1, 2, 3, 2, 1], dtype=np.int64)
    weights = np.ones_like(values, dtype=np.float32)
    axis = 0
    minlength = None
    maxlength = None
    binary_output = False
    name = None
    input_dict = {"values": values, "weights": weights, "axis": axis, "minlength": minlength, "maxlength": maxlength, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: with weights and minlength
    values = np.array([0, 1, 2], dtype=np.int64)
    weights = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    axis = 0
    minlength = 4
    maxlength = None
    binary_output = False
    name = None
    input_dict = {"values": values, "weights": weights, "axis": axis, "minlength": minlength, "maxlength": maxlength, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.bincount"] = tf_sparse_bincount_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.bincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.bincount'.")

check_valid('tf.sparse.bincount', generated_inputs['tf.sparse.bincount'], lib="tf", suffix=0)
