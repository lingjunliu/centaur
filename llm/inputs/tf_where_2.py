
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_where_inputs():
    list_of_inputs = []

    # Input 1
    condition = np.array([True, False, True])
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    name = "test_where_1"
    input_dict = {"condition": condition, "x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    condition = np.array([[True, False], [False, True]])
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    name = "test_where_2"
    input_dict = {"condition": condition, "x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    condition = np.array([False, False, False])
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    name = "test_where_3"
    input_dict = {"condition": condition, "x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    condition = np.array([True, True, True])
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    name = "test_where_4"
    input_dict = {"condition": condition, "x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    condition = np.array([[True, False], [True, True]])
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    name = "test_where_5"
    input_dict = {"condition": condition, "x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    condition = np.array([[[True, False], [True, True]], [[False, True], [False, False]]])
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    y = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    name = "test_where_6"
    input_dict = {"condition": condition, "x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    condition = np.array([True, False, True])
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    name = "test_where_7"
    input_dict = {"condition": condition, "x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    condition = np.array([[True, False], [False, True]])
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[5.0, 6.0], [7.0, 8.0]])
    name = "test_where_8"
    input_dict = {"condition": condition, "x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    condition = np.array([True, False, True], dtype=bool)
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    name = "test_where_9"
    input_dict = {"condition": condition, "x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    condition = np.array([[True, False], [False, True]], dtype=bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[5, 6], [7, 8]], dtype=np.int32)
    name = "test_where_10"
    input_dict = {"condition": condition, "x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.where_2"] = tf_where_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.where_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.where_2'.")

check_valid('tf.where', generated_inputs['tf.where_2'], lib="tf", suffix=2)
