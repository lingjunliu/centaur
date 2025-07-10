
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv1d_transpose_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = tf.constant(np.random.rand(1, 5, 3), dtype=tf.float32).numpy()
    filters_tensor = tf.constant(np.random.rand(3, 3, 3), dtype=tf.float32).numpy()
    output_shape_tensor = tf.constant([1, 7, 3], dtype=tf.int32).numpy()
    strides = [1]
    padding = 'SAME'
    data_format = 'NWC'
    dilations = [1]
    name = 'conv1d_transpose_1'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = tf.constant(np.random.rand(2, 10, 6), dtype=tf.float32).numpy()
    filters_tensor = tf.constant(np.random.rand(5, 6, 6), dtype=tf.float32).numpy()
    output_shape_tensor = tf.constant([2, 14, 6], dtype=tf.int32).numpy()
    strides = [2]
    padding = 'VALID'
    data_format = 'NWC'
    dilations = [1]
    name = 'conv1d_transpose_2'

    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "output_shape": output_shape_tensor,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.conv1d_transpose_2"] = tf_nn_conv1d_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.conv1d_transpose_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv1d_transpose_2'.")

check_valid('tf.nn.conv1d_transpose', generated_inputs['tf.nn.conv1d_transpose_2'], lib="tf", suffix=2)
