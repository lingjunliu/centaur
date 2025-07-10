
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_reverse_inputs():
    list_of_inputs = []

    # Input 1
    tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = np.array([1], dtype=np.int32)
    name = "reverse_example_1"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis = np.array([0], dtype=np.int32)
    name = "reverse_example_2"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    axis = np.array([0], dtype=np.int32)
    name = "reverse_example_3"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = np.array([[1, 2], [3, 4]], dtype=np.float64)
    axis = np.array([0, 1], dtype=np.int32)
    name = "reverse_example_4"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    axis = np.array([0, 2], dtype=np.int32)
    name = "reverse_example_5"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = np.array([-1], dtype=np.int32)
    name = "reverse_example_6"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis = np.array([-3], dtype=np.int32)
    name = "reverse_example_7"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    axis = np.array([-1], dtype=np.int32)
    name = "reverse_example_8"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = np.array([[1, 2], [3, 4]], dtype=np.float64)
    axis = np.array([-2, -1], dtype=np.int32)
    name = "reverse_example_9"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tensor = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    axis = np.array([-3, -1], dtype=np.int32)
    name = "reverse_example_10"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.reverse"] = tf_reverse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.reverse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.reverse'.")

check_valid('tf.reverse', generated_inputs['tf.reverse'], lib="tf", suffix=0)
