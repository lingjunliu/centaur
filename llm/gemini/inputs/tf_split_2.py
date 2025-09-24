
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_split_inputs():
    list_of_inputs = []

    # Input 1
    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    num_or_size_splits = [1, 2]
    axis = 1
    num = None
    name = "split_example_1"

    input_dict = {
        "value": value,
        "num_or_size_splits": num_or_size_splits,
        "axis": axis,
        "num": num,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    num_or_size_splits = [1, 1]
    axis = 0
    num = None
    name = "split_example_2"

    input_dict = {
        "value": value,
        "num_or_size_splits": num_or_size_splits,
        "axis": axis,
        "num": num,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    num_or_size_splits = [2, 4]
    axis = 0
    num = None
    name = "split_example_3"

    input_dict = {
        "value": value,
        "num_or_size_splits": num_or_size_splits,
        "axis": axis,
        "num": num,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    num_or_size_splits = [2, 2]
    axis = 1
    num = None
    name = "split_example_4"

    input_dict = {
        "value": value,
        "num_or_size_splits": num_or_size_splits,
        "axis": axis,
        "num": num,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.int32)
    num_or_size_splits = [1, 1, 1]
    axis = 0
    num = None
    name = "split_example_5"

    input_dict = {
        "value": value,
        "num_or_size_splits": num_or_size_splits,
        "axis": axis,
        "num": num,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    value = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    num_or_size_splits = [2, 1]
    axis = 0
    num = None
    name = "split_example_6"

    input_dict = {
        "value": value,
        "num_or_size_splits": num_or_size_splits,
        "axis": axis,
        "num": num,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    value = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    num_or_size_splits = [1, 1, 1, 1, 1, 1, 1, 1]
    axis = 0
    num = None
    name = "split_example_7"

    input_dict = {
        "value": value,
        "num_or_size_splits": num_or_size_splits,
        "axis": axis,
        "num": num,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    value = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    num_or_size_splits = [1, 1]
    axis = 0
    num = None
    name = "split_example_8"

    input_dict = {
        "value": value,
        "num_or_size_splits": num_or_size_splits,
        "axis": axis,
        "num": num,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    num_or_size_splits = [3]
    axis = 1
    num = None
    name = "split_example_9"

    input_dict = {
        "value": value,
        "num_or_size_splits": num_or_size_splits,
        "axis": axis,
        "num": num,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    num_or_size_splits = [2]
    axis = 2
    num = None
    name = "split_example_10"

    input_dict = {
        "value": value,
        "num_or_size_splits": num_or_size_splits,
        "axis": axis,
        "num": num,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.split_2"] = tf_split_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.split_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.split_2'.")

check_valid('tf.split', generated_inputs['tf.split_2'], lib="tf", suffix=2)
