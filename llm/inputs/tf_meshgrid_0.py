
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_meshgrid_inputs():
    list_of_inputs = []

    # Input 1: Basic example with default indexing
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    input_dict = {"args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], "indexing": "xy", "name": "mesh1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 'ij' indexing
    x = np.array([1, 2], dtype=np.float32)
    y = np.array([3, 4, 5], dtype=np.float32)
    input_dict = {"args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], "indexing": "ij", "name": "mesh2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Three dimensions
    x = np.array([1, 2], dtype=np.int64)
    y = np.array([3, 4], dtype=np.int64)
    z = np.array([5, 6], dtype=np.int64)
    input_dict = {"args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y), tf.convert_to_tensor(z)], "indexing": "xy", "name": "mesh3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    x = np.array([-1, 0, 1], dtype=np.float64)
    y = np.array([-2, -1, 0], dtype=np.float64)
    input_dict = {"args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], "indexing": "xy", "name": "mesh4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different data types (int32 and float32)
    x = np.array([1, 2], dtype=np.int32)
    y = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {"args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], "indexing": "xy", "name": "mesh5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single element arrays
    x = np.array([1], dtype=np.int32)
    y = np.array([2], dtype=np.int32)
    input_dict = {"args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], "indexing": "xy", "name": "mesh6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger arrays
    x = np.arange(10, dtype=np.float32)
    y = np.arange(5, 15, dtype=np.float32)
    input_dict = {"args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], "indexing": "xy", "name": "mesh7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 'ij' indexing with different types and name as None
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([4.0, 5.0], dtype=np.float64)
    input_dict = {"args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], "indexing": "ij", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty name
    x = np.array([1, 2], dtype=np.float32)
    y = np.array([3, 4], dtype=np.float32)
    input_dict = {"args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], "indexing": "xy", "name": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Three args with ij indexing
    x = np.array([1, 2], dtype=np.int32)
    y = np.array([3, 4], dtype=np.int32)
    z = np.array([5, 6], dtype=np.int32)
    input_dict = {"args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y), tf.convert_to_tensor(z)], "indexing": "ij", "name": "mesh10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Example with different shapes for x and y
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([5, 6], dtype=np.int32)
    input_dict = {"args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], "indexing": "xy", "name": "mesh11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Example with all the same number
    x = np.array([2, 2, 2], dtype=np.int32)
    y = np.array([2, 2, 2], dtype=np.int32)
    input_dict = {"args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], "indexing": "xy", "name": "mesh12"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: More negative example
    x = np.array([-1, -2, -3, -4], dtype=np.float64)
    y = np.array([-5, -6], dtype=np.float64)
    input_dict = {"args": [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], "indexing": "xy", "name": "mesh13"}
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
