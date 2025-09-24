
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_diag_part_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    name = "diag_1"
    k = (0, 0)
    padding_value = np.float32(0.0)
    align = "RIGHT_LEFT"
    input_dict = {"input": input_tensor, "name": name, "k": k, "padding_value": padding_value, "align": align}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    name = "diag_2"
    k = (1, 1)
    padding_value = np.float32(0.0)
    align = "LEFT_RIGHT"
    input_dict = {"input": input_tensor, "name": name, "k": k, "padding_value": padding_value, "align": align}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    name = "diag_3"
    k = (-1, -1)
    padding_value = np.float32(0.0)
    align = "RIGHT_RIGHT"
    input_dict = {"input": input_tensor, "name": name, "k": k, "padding_value": padding_value, "align": align}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    name = "diag_4"
    k = (0, 1)
    padding_value = np.float32(0.0)
    align = "LEFT_LEFT"
    input_dict = {"input": input_tensor, "name": name, "k": k, "padding_value": padding_value, "align": align}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    name = "diag_5"
    k = (-1, 0)
    padding_value = np.float32(-1.0)
    align = "RIGHT_LEFT"
    input_dict = {"input": input_tensor, "name": name, "k": k, "padding_value": padding_value, "align": align}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[1, 2], [3, 4], [5, 6]])
    name = "diag_6"
    k = (0, 0)
    padding_value = np.float32(1.0)
    align = "RIGHT_LEFT"
    input_dict = {"input": input_tensor, "name": name, "k": k, "padding_value": padding_value, "align": align}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]])
    name = "diag_7"
    k = (1, 2)
    padding_value = np.float32(9.0)
    align = "LEFT_RIGHT"
    input_dict = {"input": input_tensor, "name": name, "k": k, "padding_value": padding_value, "align": align}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    name = "diag_8"
    k = (-1, 1)
    padding_value = np.float32(-2.0)
    align = "RIGHT_RIGHT"
    input_dict = {"input": input_tensor, "name": name, "k": k, "padding_value": padding_value, "align": align}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    name = "diag_9"
    k = (0, 2)
    padding_value = np.float32(0.5)
    align = "LEFT_LEFT"
    input_dict = {"input": input_tensor, "name": name, "k": k, "padding_value": padding_value, "align": align}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10. Reduced k range to be within bounds
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    name = "diag_10"
    k = (-1, 0)
    padding_value = np.float32(3.14)
    align = "RIGHT_LEFT"
    input_dict = {"input": input_tensor, "name": name, "k": k, "padding_value": padding_value, "align": align}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.diag_part_2"] = tf_linalg_diag_part_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.diag_part_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.diag_part_2'.")

check_valid('tf.linalg.diag_part', generated_inputs['tf.linalg.diag_part_2'], lib="tf", suffix=2)
