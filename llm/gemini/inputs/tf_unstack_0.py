
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_unstack_inputs():
    list_of_inputs = []

    # Input 1
    value = np.array([[1, 2, 3], [4, 5, 6]])
    num = None
    axis = 0
    name = "unstack_0"
    input_dict = {"value": value, "num": num, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.array([[1, 2, 3], [4, 5, 6]])
    num = None
    axis = 1
    name = "unstack_1"
    input_dict = {"value": value, "num": num, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    num = None
    axis = 0
    name = "unstack_2"
    input_dict = {"value": value, "num": num, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    num = None
    axis = 1
    name = "unstack_3"
    input_dict = {"value": value, "num": num, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    num = None
    axis = 2
    name = "unstack_4"
    input_dict = {"value": value, "num": num, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    value = np.array([[1, 2, 3], [4, 5, 6]])
    num = 3
    axis = 1
    name = "unstack_5"
    input_dict = {"value": value, "num": num, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    value = np.array([1, 2, 3, 4])
    num = None
    axis = 0
    name = "unstack_6"
    input_dict = {"value": value, "num": num, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    value = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]])
    num = None
    axis = 2
    name = "unstack_7"
    input_dict = {"value": value, "num": num, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Test negative axis
    value = np.array([[1, 2, 3], [4, 5, 6]])
    num = None
    axis = -1
    name = "unstack_8"
    input_dict = {"value": value, "num": num, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    num = 2
    axis = 0
    name = "unstack_9"
    input_dict = {"value": value, "num": num, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.unstack"] = tf_unstack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.unstack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.unstack'.")

check_valid('tf.unstack', generated_inputs['tf.unstack'], lib="tf", suffix=0)
