
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_squeeze_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[1], [2], [3]]])
    axis = []
    name = "squeeze_example_1"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[1, 2, 3]]])
    axis = [1]
    name = "squeeze_example_2"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([1, 2, 3])
    axis = []
    name = "squeeze_example_3"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[[1]]], [[[2]]]])
    axis = [2, 3]
    name = "squeeze_example_4"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_tensor = np.array([[[[[1]]]]])
    axis = [0, 1, 2, 3]
    name = "squeeze_example_5"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1], [2]]])
    axis = [2]
    name = "squeeze_example_6"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[1, 2, 3]]])
    axis = []
    name = "squeeze_example_7"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[[[1]]]]])
    axis = []
    name = "squeeze_example_8"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[1]])
    axis = [1]
    name = "squeeze_example_9"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[1]]])
    axis = [0]
    name = "squeeze_example_10"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_tensor = np.array([[[1], [2]]])
    axis = []
    name = "squeeze_example_11"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.squeeze"] = tf_squeeze_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.squeeze' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.squeeze'.")

check_valid('tf.squeeze', generated_inputs['tf.squeeze'], lib="tf", suffix=0)
