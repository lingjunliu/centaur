
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_meshgrid_inputs():
    list_of_inputs = []

    # Input 1: Basic example with xy indexing
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    input_dict = {'args': [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], 'indexing': 'xy', 'name': 'meshgrid_xy'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic example with ij indexing
    x = np.array([1, 2, 3], dtype=np.float32)
    y = np.array([4, 5, 6], dtype=np.float32)
    input_dict = {'args': [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], 'indexing': 'ij', 'name': 'meshgrid_ij'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different data types (float64) and xy indexing
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([4.0, 5.0], dtype=np.float64)
    input_dict = {'args': [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], 'indexing': 'xy', 'name': 'meshgrid_xy_float64'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data types (int64) and ij indexing
    x = np.array([1, 2], dtype=np.int64)
    y = np.array([4, 5, 6], dtype=np.int64)
    input_dict = {'args': [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], 'indexing': 'ij', 'name': 'meshgrid_ij_int64'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Three input arrays with xy indexing
    x = np.array([1, 2], dtype=np.int32)
    y = np.array([3, 4], dtype=np.int32)
    z = np.array([5, 6], dtype=np.int32)
    input_dict = {'args': [tf.convert_to_tensor(x), tf.convert_to_tensor(y), tf.convert_to_tensor(z)], 'indexing': 'xy', 'name': 'meshgrid_3d_xy'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Three input arrays with ij indexing
    x = np.array([1, 2], dtype=np.float32)
    y = np.array([3, 4], dtype=np.float32)
    z = np.array([5, 6], dtype=np.float32)
    input_dict = {'args': [tf.convert_to_tensor(x), tf.convert_to_tensor(y), tf.convert_to_tensor(z)], 'indexing': 'ij', 'name': 'meshgrid_3d_ij'}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Different shapes, xy indexing
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([5, 6], dtype=np.int32)
    input_dict = {'args': [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], 'indexing': 'xy', 'name': 'meshgrid_xy_diff_shape'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different shapes, ij indexing
    x = np.array([1, 2], dtype=np.float32)
    y = np.array([3, 4, 5], dtype=np.float32)
    input_dict = {'args': [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], 'indexing': 'ij', 'name': 'meshgrid_ij_diff_shape'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single element arrays, xy indexing
    x = np.array([1], dtype=np.int32)
    y = np.array([2], dtype=np.int32)
    input_dict = {'args': [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], 'indexing': 'xy', 'name': 'meshgrid_xy_single'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single element arrays, ij indexing
    x = np.array([1.0], dtype=np.float32)
    y = np.array([2.0], dtype=np.float32)
    input_dict = {'args': [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], 'indexing': 'ij', 'name': 'meshgrid_ij_single'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative values
    x = np.array([-1, 0, 1], dtype=np.int32)
    y = np.array([-2, 2], dtype=np.int32)
    input_dict = {'args': [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], 'indexing': 'xy', 'name': 'meshgrid_xy_negative'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: All negative values
    x = np.array([-3, -2, -1], dtype=np.float32)
    y = np.array([-5, -4], dtype=np.float32)
    input_dict = {'args': [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], 'indexing': 'ij', 'name': 'meshgrid_ij_all_negative'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Empty name
    x = np.array([1, 2], dtype=np.int32)
    y = np.array([3, 4], dtype=np.int32)
    input_dict = {'args': [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], 'indexing': 'xy', 'name': ''}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14: Whitespace name
    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([3.0, 4.0], dtype=np.float32)
    input_dict = {'args': [tf.convert_to_tensor(x), tf.convert_to_tensor(y)], 'indexing': 'ij', 'name': '   '}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    #Fix: The error occurs because the shape attribute is being accessed from a list. The 'args' value is a list of tensors.
    #The 'args' key in the dictionary holds a list of TF tensors, not a single tensor. The get_ll function in input_generators.py expects a single tensor to extract the shape from.
    #So, we can not pass the list of tensors directly to get_ll method. Since we have a list of tensors, it is better to iterate through the list

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
