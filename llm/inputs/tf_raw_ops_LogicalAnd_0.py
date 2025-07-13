
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_logical_and_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([True, False, True])
    y = np.array([False, True, True])
    name = None
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array(True)
    y = np.array(False)
    name = "and_op"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[True, False], [False, True]])
    y = np.array([[False, True], [False, False]])
    name = None
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([True, False])
    y = np.array(True)
    name = None
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array(False)
    y = np.array([True, True, False])
    name = "broadcast_and"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    y = np.array([[[False, True], [True, False]], [[True, False], [False, True]]])
    name = None
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting case
    x = np.array([[True, False]])
    y = np.array([[True], [False]])
    name = None
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([True])
    y = np.array([False])
    name = None
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([False, False, True, True])
    y = np.array([False, True, False, True])
    name = None
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[[True, False], [False, True]]])
    y = np.array([[[False, True], [True, False]]])
    name = "logical_and_3d"
    input_dict = {"x": x, "y": y, "name": name}
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
