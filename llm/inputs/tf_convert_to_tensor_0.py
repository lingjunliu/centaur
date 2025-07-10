
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_convert_to_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic list of integers
    value = [1, 2, 3, 4, 5]
    dtype = tf.int32
    dtype_hint = tf.float32
    name = "int_list"
    input_dict = {"value": value, "dtype": dtype, "dtype_hint": dtype_hint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of floats
    value = [1.0, 2.5, 3.7, 4.2, 5.9]
    dtype = tf.float32
    dtype_hint = tf.int32
    name = "float_list"
    input_dict = {"value": value, "dtype": dtype, "dtype_hint": dtype_hint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of strings (removed due to numpy error)
    #value = ["hello", "world", "tensorflow"]
    #dtype = tf.string
    #dtype_hint = tf.int32
    #name = "string_list"
    #input_dict = {"value": value, "dtype": dtype, "dtype_hint": dtype_hint, "name": name}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of mixed numbers
    value = [1, 2.5, 3, 4.7, 5]
    dtype = tf.float32
    dtype_hint = tf.int32
    name = "mixed_list"
    input_dict = {"value": value, "dtype": dtype, "dtype_hint": dtype_hint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List of lists (2D array)
    value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    dtype = tf.int32
    dtype_hint = tf.float32
    name = "2d_list"
    input_dict = {"value": value, "dtype": dtype, "dtype_hint": dtype_hint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List of lists with same lengths
    value = [[1, 2], [3, 4], [5,6]]
    dtype = tf.int32
    dtype_hint = tf.float32
    name = "list_same_len"
    input_dict = {"value": value, "dtype": dtype, "dtype_hint": dtype_hint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 7: List of booleans
    value = [True, False, True, True, False]
    dtype = tf.bool
    dtype_hint = tf.int32
    name = "bool_list"
    input_dict = {"value": value, "dtype": dtype, "dtype_hint": dtype_hint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))



    # Input 8: List with negative values
    value = [-1, -2, 0, 1, 2]
    dtype = tf.int32
    dtype_hint = tf.float32
    name = "negative_list"
    input_dict = {"value": value, "dtype": dtype, "dtype_hint": dtype_hint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty list
    value = []
    dtype = tf.int32
    dtype_hint = tf.float32
    name = "empty_list"
    input_dict = {"value": value, "dtype": dtype, "dtype_hint": dtype_hint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: List of large integers
    value = [2**10 , 2**9, 2**8]
    dtype = tf.int32
    dtype_hint = tf.int32
    name = "large_int_list"
    input_dict = {"value": value, "dtype": dtype, "dtype_hint": dtype_hint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.convert_to_tensor"] = tf_convert_to_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.convert_to_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.convert_to_tensor'.")

check_valid('tf.convert_to_tensor', generated_inputs['tf.convert_to_tensor'], lib="tf", suffix=0)
