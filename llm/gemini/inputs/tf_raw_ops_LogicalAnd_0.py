
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_logical_and_inputs():
    list_of_inputs = []

    # Input 1: Basic case, two bool tensors
    x = np.array([True, False, True], dtype=bool)
    y = np.array([False, True, True], dtype=bool)
    input_dict = {"x": x, "y": y, "name": "logical_and_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcasting, single element vs tensor
    x = np.array(True, dtype=bool)
    y = np.array([False, True, False], dtype=bool)
    input_dict = {"x": x, "y": y, "name": "logical_and_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasting, tensor vs single element
    x = np.array([False, True, False], dtype=bool)
    y = np.array(True, dtype=bool)
    input_dict = {"x": x, "y": y, "name": "logical_and_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Two tensors of different shapes (broadcasting)
    x = np.array([[True, False], [False, True]], dtype=bool)
    y = np.array([True, False], dtype=bool)
    input_dict = {"x": x, "y": y, "name": "logical_and_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Two tensors of different shapes (broadcasting) - different arrangement
    x = np.array([True, False], dtype=bool)
    y = np.array([[True], [False]], dtype=bool)
    input_dict = {"x": x, "y": y, "name": "logical_and_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Two tensors, all False
    x = np.array([False, False, False], dtype=bool)
    y = np.array([False, False, False], dtype=bool)
    input_dict = {"x": x, "y": y, "name": "logical_and_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Two tensors, all True
    x = np.array([True, True, True], dtype=bool)
    y = np.array([True, True, True], dtype=bool)
    input_dict = {"x": x, "y": y, "name": "logical_and_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Empty tensors
    x = np.array([], dtype=bool)
    y = np.array([], dtype=bool)
    input_dict = {"x": x, "y": y, "name": "logical_and_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multi-dimensional tensors
    x = np.array([[[True, False], [False, True]], [[False, True], [True, False]]], dtype=bool)
    y = np.array([[[False, True], [True, False]], [[True, False], [False, True]]], dtype=bool)
    input_dict = {"x": x, "y": y, "name": "logical_and_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Single element boolean values
    x = np.array(True, dtype=bool)
    y = np.array(False, dtype=bool)
    input_dict = {"x": x, "y": y, "name": "logical_and_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.LogicalAnd"] = tf_raw_ops_logical_and_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.LogicalAnd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LogicalAnd'.")

check_valid('tf.raw_ops.LogicalAnd', generated_inputs['tf.raw_ops.LogicalAnd'], lib="tf", suffix=0)
