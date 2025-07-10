
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_meshgrid_inputs():
    list_of_inputs = []

    # Input 1: Basic example with xy indexing
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)],
        "indexing": "xy",
        "name": "meshgrid_xy"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic example with ij indexing
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)],
        "indexing": "ij",
        "name": "meshgrid_ij"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Three dimensions with xy indexing, different sizes
    x = np.array([1, 2])
    y = np.array([3, 4, 5])
    z = np.array([6])
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y), tf.convert_to_tensor(z)],
        "indexing": "xy",
        "name": "meshgrid_3d_xy_diff_size"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Three dimensions with ij indexing, different sizes
    x = np.array([1, 2])
    y = np.array([3, 4, 5])
    z = np.array([6])
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y), tf.convert_to_tensor(z)],
        "indexing": "ij",
        "name": "meshgrid_3d_ij_diff_size"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: empty array
    x = np.array([])
    y = np.array([])
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)],
        "indexing": "xy",
        "name": "meshgrid_empty_xy"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: empty array, ij
    x = np.array([])
    y = np.array([])
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)],
        "indexing": "ij",
        "name": "meshgrid_empty_ij"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: xy, one is empty
    x = np.array([1, 2, 3])
    y = np.array([])
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)],
        "indexing": "xy",
        "name": "meshgrid_one_empty_xy"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: ij, one is empty
    x = np.array([1, 2, 3])
    y = np.array([])
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)],
        "indexing": "ij",
        "name": "meshgrid_one_empty_ij"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger arrays with xy indexing
    x = np.arange(10)
    y = np.arange(5, 15)
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)],
        "indexing": "xy",
        "name": "meshgrid_large_xy"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger arrays with ij indexing
    x = np.arange(10)
    y = np.arange(5, 15)
    input_dict = {
        "args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)],
        "indexing": "ij",
        "name": "meshgrid_large_ij"
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
