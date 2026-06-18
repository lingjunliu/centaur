
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_atrous_conv2d_inputs():
    list_of_inputs = []
    
    # Input 1
    value = np.random.randn(1, 3, 3, 1).astype(np.float32)
    filters = np.random.randn(2, 2, 1, 1).astype(np.float32)
    rate = 1
    padding = 'VALID'
    name = 'atrous_conv_1'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 2
    value = np.random.randn(2, 5, 5, 3).astype(np.float32)
    filters = np.random.randn(3, 3, 3, 4).astype(np.float32)
    rate = 2
    padding = 'SAME'
    name = 'atrous_conv_2'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 3
    value = np.random.randn(1, 7, 7, 2).astype(np.float64)
    filters = np.random.randn(3, 3, 2, 2).astype(np.float64)
    rate = 3
    padding = 'VALID'
    name = 'atrous_conv_3'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 4
    value = np.random.randn(4, 10, 10, 8).astype(np.float32)
    filters = np.random.randn(1, 1, 8, 16).astype(np.float32)
    rate = 1
    padding = 'SAME'
    name = 'atrous_conv_4'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 5
    value = np.random.randn(2, 8, 12, 3).astype(np.float32)
    filters = np.random.randn(3, 5, 3, 4).astype(np.float32)
    rate = 2
    padding = 'VALID'
    name = 'atrous_conv_5'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 6
    value = np.random.randn(1, 14, 14, 1).astype(np.float32)
    filters = np.random.randn(5, 5, 1, 2).astype(np.float32)
    rate = 4
    padding = 'SAME'
    name = 'atrous_conv_6'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 7
    value = np.random.randn(1, 20, 20, 4).astype(np.float32)
    filters = np.random.randn(2, 2, 4, 4).astype(np.float32)
    rate = 5
    padding = 'SAME'
    name = 'atrous_conv_7'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 8
    value = np.random.randn(3, 6, 6, 16).astype(np.float32)
    filters = np.random.randn(1, 1, 16, 8).astype(np.float32)
    rate = 2
    padding = 'VALID'
    name = 'atrous_conv_8'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 9
    value = np.random.randn(1, 5, 5, 32).astype(np.float32)
    filters = np.random.randn(3, 3, 32, 64).astype(np.float32)
    rate = 1
    padding = 'SAME'
    name = 'atrous_conv_9'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 10
    value = np.random.randn(2, 15, 15, 3).astype(np.float32)
    filters = np.random.randn(3, 3, 3, 3).astype(np.float32)
    rate = 3
    padding = 'VALID'
    name = 'atrous_conv_10'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    return list_of_inputs

generated_inputs["tf.nn.atrous_conv2d"] = tf_nn_atrous_conv2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.atrous_conv2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.atrous_conv2d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.atrous_conv2d', generated_inputs['tf.nn.atrous_conv2d'], lib="tf", suffix=0)
