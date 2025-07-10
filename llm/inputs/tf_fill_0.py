
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_fill_inputs():
    list_of_inputs = []

    # Input 1: Simple case with integers
    dims = [2, 3]
    value = tf.constant(5)
    name = "fill_example_1"
    layout = None

    input_dict = {
        "dims": dims,
        "value": value,
        "name": name,
        "layout": layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different dimensions and data type
    dims = [1, 4, 2]
    value = tf.constant(3.14, dtype=tf.float32)
    name = "fill_example_2"
    layout = None

    input_dict = {
        "dims": dims,
        "value": value,
        "name": name,
        "layout": layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero dimension
    dims = [0, 5]
    value = tf.constant(10)
    name = "fill_example_3"
    layout = None

    input_dict = {
        "dims": dims,
        "value": value,
        "name": name,
        "layout": layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger dimensions
    dims = [10, 10]
    value = tf.constant(-1, dtype=tf.int32)
    name = "fill_example_4"
    layout = None

    input_dict = {
        "dims": dims,
        "value": value,
        "name": name,
        "layout": layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String value
    dims = [2, 2]
    value = tf.constant("hello", dtype=tf.string)
    name = "fill_example_5"
    layout = None

    input_dict = {
        "dims": dims,
        "value": value,
        "name": name,
        "layout": layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boolean value
    dims = [3, 3]
    value = tf.constant(True, dtype=tf.bool)
    name = "fill_example_6"
    layout = None

    input_dict = {
        "dims": dims,
        "value": value,
        "name": name,
        "layout": layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: One dimensional tensor
    dims = [7]
    value = tf.constant(77)
    name = "fill_example_7"
    layout = None

    input_dict = {
        "dims": dims,
        "value": value,
        "name": name,
        "layout": layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large integer
    dims = [1, 2]
    value = tf.constant(2**31 - 1, dtype=tf.int32)
    name = "fill_example_8"
    layout = None

    input_dict = {
        "dims": dims,
        "value": value,
        "name": name,
        "layout": layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative float
    dims = [2, 1]
    value = tf.constant(-2.71, dtype=tf.float32)
    name = "fill_example_9"
    layout = None

    input_dict = {
        "dims": dims,
        "value": value,
        "name": name,
        "layout": layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Higher rank tensor
    dims = [1, 1, 1, 1]
    value = tf.constant(1)
    name = "fill_example_10"
    layout = None
    
    input_dict = {
        "dims": dims,
        "value": value,
        "name": name,
        "layout": layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.fill"] = tf_fill_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.fill' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.fill'.")

check_valid('tf.fill', generated_inputs['tf.fill'], lib="tf", suffix=0)
