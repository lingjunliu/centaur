
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_logical_not_inputs():
    list_of_inputs = []

    # Input 1: Simple boolean tensor
    x = np.array([True, False, True])
    name = "logical_not_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D boolean tensor
    x = np.array([[True, False], [False, True]])
    name = "logical_not_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D boolean tensor
    x = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    name = "logical_not_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty boolean tensor
    x = np.array([])
    name = "logical_not_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single element boolean tensor
    x = np.array([True])
    name = "logical_not_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All True values
    x = np.array([True, True, True])
    name = "logical_not_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All False values
    x = np.array([False, False, False])
    name = "logical_not_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mixed values, different name
    x = np.array([True, False, True, False])
    name = "another_name"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger tensor
    x = np.random.choice([True, False], size=(10, 10))
    name = "logical_not_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another 3D tensor with larger dimensions
    x = np.random.choice([True, False], size=(3, 4, 5))
    name = "logical_not_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.LogicalNot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LogicalNot'.")

check_valid('tf.raw_ops.LogicalNot', generated_inputs['tf.raw_ops.LogicalNot'], lib="tf", suffix=0)
