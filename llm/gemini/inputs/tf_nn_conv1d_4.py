
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv1d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 NWC valid combination with 'SAME' padding
    input_val = np.random.randn(2, 10, 3).astype(np.float32)
    filters_val = np.random.randn(3, 3, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NWC format with float32, stride [2], and 'VALID' padding
    input_val = np.random.randn(2, 10, 3).astype(np.float32)
    filters_val = np.random.randn(3, 3, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [2],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NWC format, float16 with dilation 2
    input_val = np.random.randn(1, 8, 2).astype(np.float16)
    filters_val = np.random.randn(2, 2, 2).astype(np.float16)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 2,
        'name': 'conv1d_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NWC format, float64, standard parameters
    input_val = np.random.randn(4, 16, 4).astype(np.float64)
    filters_val = np.random.randn(3, 4, 8).astype(np.float64)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NWC format with negative values in inputs
    input_val = np.array([[[-1.0, -2.0], [3.0, 4.0], [-5.0, -6.0]]], dtype=np.float32)
    filters_val = np.array([[[1.0, -1.0], [-1.0, 1.0]], [[2.0, -2.0], [-2.0, 2.0]]], dtype=np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [1],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher dimension batch (batch_shape = [2, 3]), NWC format
    input_val = np.random.randn(2, 3, 5, 2).astype(np.float32)
    filters_val = np.random.randn(2, 2, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Higher dimension batch (batch_shape = [2, 2]), NWC format
    input_val = np.random.randn(2, 2, 8, 3).astype(np.float32)
    filters_val = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [2],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger stride [3]
    input_val = np.random.randn(1, 15, 3).astype(np.float32)
    filters_val = np.random.randn(3, 3, 3).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [3],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Dilation of 3
    input_val = np.random.randn(1, 12, 2).astype(np.float32)
    filters_val = np.random.randn(3, 2, 2).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [1],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 3,
        'name': 'conv1d_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger channel sizes
    input_val = np.random.randn(1, 5, 64).astype(np.float32)
    filters_val = np.random.randn(3, 64, 32).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.conv1d_4"] = tf_nn_conv1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.conv1d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv1d_4'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.conv1d', generated_inputs['tf.nn.conv1d_4'], lib="tf", suffix=4)
