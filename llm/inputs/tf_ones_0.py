
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ones_inputs():
    list_of_inputs = []

    # Input 1: Basic usage with int shape
    shape = [2, 3]
    dtype = tf.int32
    name = "ones_tensor_1"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic usage with float shape
    shape = [4, 2]
    dtype = tf.float32
    name = "ones_tensor_2"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different dtype
    shape = [1, 5]
    dtype = tf.bool
    name = "ones_tensor_3"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor
    shape = [2, 2, 2]
    dtype = tf.float64
    name = "ones_tensor_4"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different name
    shape = [3, 1]
    dtype = tf.complex64
    name = "another_name"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty shape
    shape = []
    dtype = tf.float32
    name = "empty_shape"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Shape as a list
    shape = [2, 4]
    dtype = tf.int8
    name = "shape_as_tensor"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large shape
    shape = [100, 100]
    dtype = tf.float32
    name = "large_shape"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D tensor
    shape = [2, 3, 4, 5]
    dtype = tf.float32
    name = "4d_tensor"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10:  Complex128 type
    shape = [2, 2]
    dtype = tf.complex128
    name = "complex128_tensor"
    layout = None
    input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Layout specified, remove since tensors can't be converted
    # shape = [2, 2]
    # dtype = tf.float32
    # name = "layout_specified"
    # layout = tf.zeros([2, 2])
    # input_dict = {"shape": shape, "dtype": dtype, "name": name, "layout": layout}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ones"] = tf_ones_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ones' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ones'.")

check_valid('tf.ones', generated_inputs['tf.ones'], lib="tf", suffix=0)
