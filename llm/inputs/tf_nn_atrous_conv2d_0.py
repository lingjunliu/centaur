
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_atrous_conv2d_inputs():
    list_of_inputs = []

    # Input 1
    value = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filters = np.random.rand(3, 3, 3, 16).astype(np.float32)
    rate = 2
    padding = 'SAME'
    name = 'atrous_conv1'
    input_dict = {'value': value, 'filters': filters, 'rate': rate, 'padding': padding, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.random.rand(2, 20, 20, 1).astype(np.float32)
    filters = np.random.rand(5, 5, 1, 8).astype(np.float32)
    rate = 1
    padding = 'VALID'
    name = 'atrous_conv2'
    input_dict = {'value': value, 'filters': filters, 'rate': rate, 'padding': padding, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.random.rand(4, 15, 15, 32).astype(np.float32)
    filters = np.random.rand(1, 1, 32, 64).astype(np.float32)
    rate = 3
    padding = 'SAME'
    name = 'atrous_conv3'
    input_dict = {'value': value, 'filters': filters, 'rate': rate, 'padding': padding, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.random.rand(1, 8, 8, 64).astype(np.float32)
    filters = np.random.rand(3, 3, 64, 128).astype(np.float32)
    rate = 1
    padding = 'VALID'
    name = 'atrous_conv4'
    input_dict = {'value': value, 'filters': filters, 'rate': rate, 'padding': padding, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.random.rand(2, 12, 12, 16).astype(np.float32)
    filters = np.random.rand(2, 2, 16, 32).astype(np.float32)
    rate = 5
    padding = 'SAME'
    name = 'atrous_conv5'
    input_dict = {'value': value, 'filters': filters, 'rate': rate, 'padding': padding, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    value = np.random.rand(1, 16, 16, 4).astype(np.float32)
    filters = np.random.rand(4, 4, 4, 8).astype(np.float32)
    rate = 1
    padding = 'VALID'
    name = 'atrous_conv6'
    input_dict = {'value': value, 'filters': filters, 'rate': rate, 'padding': padding, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    value = np.random.rand(4, 24, 24, 8).astype(np.float32)
    filters = np.random.rand(3, 3, 8, 16).astype(np.float32)
    rate = 2
    padding = 'SAME'
    name = 'atrous_conv7'
    input_dict = {'value': value, 'filters': filters, 'rate': rate, 'padding': padding, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    value = np.random.rand(2, 32, 32, 16).astype(np.float32)
    filters = np.random.rand(5, 5, 16, 32).astype(np.float32)
    rate = 3
    padding = 'VALID'
    name = 'atrous_conv8'
    input_dict = {'value': value, 'filters': filters, 'rate': rate, 'padding': padding, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    value = np.random.rand(1, 40, 40, 32).astype(np.float32)
    filters = np.random.rand(1, 1, 32, 64).astype(np.float32)
    rate = 4
    padding = 'SAME'
    name = 'atrous_conv9'
    input_dict = {'value': value, 'filters': filters, 'rate': rate, 'padding': padding, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.random.rand(2, 48, 48, 64).astype(np.float32)
    filters = np.random.rand(7, 7, 64, 128).astype(np.float32)
    rate = 1
    padding = 'VALID'
    name = 'atrous_conv10'
    input_dict = {'value': value, 'filters': filters, 'rate': rate, 'padding': padding, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.atrous_conv2d"] = tf_nn_atrous_conv2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.atrous_conv2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.atrous_conv2d'.")

check_valid('tf.nn.atrous_conv2d', generated_inputs['tf.nn.atrous_conv2d'], lib="tf", suffix=0)
