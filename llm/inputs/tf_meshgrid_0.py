
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_meshgrid_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)],
        "indexing": "xy",
        "name": "meshgrid_example_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([1, 2], dtype=np.float32)
    y = np.array([3, 4, 5], dtype=np.float32)
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)],
        "indexing": "ij",
        "name": "meshgrid_example_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([-1, 0, 1], dtype=np.int64)
    y = np.array([-2, 2], dtype=np.int64)
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)],
        "indexing": "xy",
        "name": "meshgrid_example_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    y = np.array([0.4, 0.5], dtype=np.float64)
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)],
        "indexing": "ij",
        "name": "meshgrid_example_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1, 2], dtype=np.int32)
    y = np.array([3, 4], dtype=np.int32)
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)],
        "indexing": "xy",
        "name": "meshgrid_example_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([1, 2], dtype=np.int32)
    y = np.array([3, 4], dtype=np.int32)
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)],
        "indexing": "ij",
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([1, 2, 3], dtype=np.int16)
    y = np.array([4, 5], dtype=np.uint8)
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)],
        "indexing": "xy",
        "name": "meshgrid_example_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.meshgrid"] = tf_meshgrid_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.meshgrid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.meshgrid'.")

check_valid('tf.meshgrid', generated_inputs['tf.meshgrid'], lib="tf", suffix=0)
