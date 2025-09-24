
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_logical_not_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([True, False], dtype=np.bool_)
    input_dict = {"x": x.astype(np.bool_), "name": "logical_not_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    input_dict = {"x": x.astype(np.bool_), "name": "logical_not_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[[True, False], [False, True]], [[False, True], [True, False]]], dtype=np.bool_)
    input_dict = {"x": x.astype(np.bool_), "name": "logical_not_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([True], dtype=np.bool_)
    input_dict = {"x": x.astype(np.bool_), "name": "logical_not_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([False], dtype=np.bool_)
    input_dict = {"x": x.astype(np.bool_), "name": "logical_not_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([True, True, True], dtype=np.bool_)
    input_dict = {"x": x.astype(np.bool_), "name": "logical_not_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([False, False, False], dtype=np.bool_)
    input_dict = {"x": x.astype(np.bool_), "name": "logical_not_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([[True, True, False], [False, True, True]], dtype=np.bool_)
    input_dict = {"x": x.astype(np.bool_), "name": "logical_not_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array(True, dtype=np.bool_)
    input_dict = {"x": x.astype(np.bool_), "name": "logical_not_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array(False, dtype=np.bool_)
    input_dict = {"x": x.astype(np.bool_), "name": "logical_not_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.LogicalNot"] = tf_raw_ops_logical_not_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.LogicalNot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LogicalNot'.")

check_valid('tf.raw_ops.LogicalNot', generated_inputs['tf.raw_ops.LogicalNot'], lib="tf", suffix=0)
