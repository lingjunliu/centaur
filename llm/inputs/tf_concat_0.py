
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_concat_inputs():
    list_of_inputs = []

    def create_input(values, axis, name):
        return {"values": values, "axis": axis, "name": name}

    # Input 1
    values = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])]
    list_of_inputs.append(copy.deepcopy(create_input(values, 0, "concat_example_1")))

    # Input 2
    values = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])]
    list_of_inputs.append(copy.deepcopy(create_input(values, 1, "concat_example_2")))

    # Input 3
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    list_of_inputs.append(copy.deepcopy(create_input(values, 0, "concat_example_3")))

    # Input 4
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    list_of_inputs.append(copy.deepcopy(create_input(values, 1, "concat_example_4")))

    # Input 5
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    list_of_inputs.append(copy.deepcopy(create_input(values, 2, "concat_example_5")))

    # Input 6: Negative axis
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    list_of_inputs.append(copy.deepcopy(create_input(values, -1, "concat_example_6")))

    # Input 7: Single tensor
    values = [np.array([[1, 2, 3], [4, 5, 6]])]
    list_of_inputs.append(copy.deepcopy(create_input(values, 0, "concat_example_7")))

    # Input 8: Negative axis with different rank
    values = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    list_of_inputs.append(copy.deepcopy(create_input(values, -1, "concat_example_8")))

    # Input 9: Four Dimensions
    values = [np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]),
              np.array([[[[17, 18], [19, 20]], [[21, 22], [23, 24]]], [[[25, 26], [27, 28]], [[29, 30], [31, 32]]]])]
    list_of_inputs.append(copy.deepcopy(create_input(values, 3, "concat_example_9")))

    return list_of_inputs

generated_inputs["tf.concat"] = tf_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.concat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.concat'.")

check_valid('tf.concat', generated_inputs['tf.concat'], lib="tf", suffix=0)
