
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_stack_inputs():
    list_of_inputs = []

    # Input 1
    values = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    axis = 0
    name = "stack_example_1"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    values = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    axis = 1
    name = "stack_example_2"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    values = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    axis = 0
    name = "stack_example_3"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    values = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    axis = 1
    name = "stack_example_4"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    values = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    axis = -1
    name = "stack_example_5"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    axis = 0
    name = "stack_example_6"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    values = [np.array([1, 2, 3])]
    axis = 0
    name = "stack_example_7"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    values = [np.array([[1, 2, 3]])]
    axis = 1
    name = "stack_example_8"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    values = [np.array([1, 2]), np.array([3, 4])]
    axis = 0
    name = "stack_example_9"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    values = [np.array([1, 2, 3], dtype=np.int64), np.array([4, 5, 6], dtype=np.int64), np.array([7, 8, 9], dtype=np.int64)]
    axis = 0
    name = "stack_example_10"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    values = [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32)]
    axis = -1
    name = "stack_example_11"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
input_list = tf_stack_inputs()
for i in range(len(input_list)):
    values = input_list[i]['values']
    input_list[i]['values'] = [np.asarray(v) for v in values]
    input_list[i]['axis'] = int(input_list[i]['axis'])

generated_inputs["tf.stack"] = input_list

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.stack'.")

check_valid('tf.stack', generated_inputs['tf.stack'], lib="tf", suffix=0)
