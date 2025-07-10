
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_full_inputs():
    list_of_inputs = []

    # Input 1
    shape = [2, 3]
    fill_value = tf.constant(5)
    dtype = tf.int32
    input_dict = {"shape": shape, "fill_value": fill_value, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = [1, 4, 2]
    fill_value = tf.constant(3.14)
    dtype = tf.float32
    input_dict = {"shape": shape, "fill_value": fill_value, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = [5]
    fill_value = tf.constant(-1)
    dtype = tf.int64
    input_dict = {"shape": shape, "fill_value": fill_value, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = [2, 2, 2, 2]
    fill_value = tf.constant(True)
    dtype = tf.bool
    input_dict = {"shape": shape, "fill_value": fill_value, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = [3, 1]
    fill_value = tf.constant(2.718, dtype=tf.float64)
    dtype = tf.float64
    input_dict = {"shape": shape, "fill_value": fill_value, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = []
    fill_value = tf.constant(10)
    dtype = tf.int32
    input_dict = {"shape": shape, "fill_value": fill_value, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = [5, 5]
    fill_value = tf.constant(-2.5)
    dtype = tf.float32
    input_dict = {"shape": shape, "fill_value": fill_value, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = [10]
    fill_value = tf.constant(0, dtype=tf.int8)
    dtype = tf.int8
    input_dict = {"shape": shape, "fill_value": fill_value, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    shape = [2, 3, 4]
    fill_value = tf.constant(1.618, dtype=tf.float16)
    dtype = tf.float16
    input_dict = {"shape": shape, "fill_value": fill_value, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = [1, 1, 1, 1, 1]
    fill_value = tf.constant(False, dtype=tf.bool)
    dtype = tf.bool
    input_dict = {"shape": shape, "fill_value": fill_value, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.full"] = tf_experimental_numpy_full_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.full' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.full'.")

check_valid('tf.experimental.numpy.full', generated_inputs['tf.experimental.numpy.full'], lib="tf", suffix=0)
