
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_zeros_inputs():
    list_of_inputs = []

    # Input 1
    shape = [2, 3]
    dtype = tf.int32
    name = "zeros_tensor_1"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = [5]
    dtype = tf.float64
    name = "zeros_tensor_2"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = [1, 4, 2]
    dtype = tf.bool
    name = "zeros_tensor_3"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = [0]
    dtype = tf.float32
    name = "zeros_tensor_4"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = [2, 2, 2, 2]
    dtype = tf.complex64
    name = "zeros_tensor_5"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = [1]
    dtype = tf.string
    name = "zeros_tensor_6"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = [10]
    dtype = tf.uint8
    name = "zeros_tensor_7"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = [1, 1, 1, 1, 1]
    dtype = tf.int16
    name = "zeros_tensor_8"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = [7, 1]
    dtype = tf.bfloat16
    name = "zeros_tensor_9"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    shape = [3, 4]
    dtype = tf.float32
    name = "zeros_tensor_10"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.zeros"] = tf_zeros_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.zeros' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.zeros'.")

check_valid('tf.zeros', generated_inputs['tf.zeros'], lib="tf", suffix=0)
