
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_conv1d_inputs():
    list_of_inputs = []
    
    # Input 1: NWC, float32, stride 1, padding SAME, dilation 1
    input_dict = {
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 5).astype(np.float32),
        'stride': 1,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: NWC, float32, stride 2, padding VALID, dilation 1
    input_dict = {
        'input': np.random.randn(1, 15, 4).astype(np.float32),
        'filters': np.random.randn(2, 4, 2).astype(np.float32),
        'stride': 2,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: NWC, float64, stride 1, padding SAME, dilation 2
    input_dict = {
        'input': np.random.randn(4, 8, 2).astype(np.float64),
        'filters': np.random.randn(3, 2, 4).astype(np.float64),
        'stride': 1,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 2,
        'name': 'conv1d_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: NWC, float32, stride 2, padding VALID, dilation 2
    input_dict = {
        'input': np.random.randn(2, 12, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 3).astype(np.float32),
        'stride': 2,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 2,
        'name': 'conv1d_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: NWC, float32, stride 3, padding SAME, dilation 1
    input_dict = {
        'input': np.random.randn(3, 20, 1).astype(np.float32),
        'filters': np.random.randn(5, 1, 8).astype(np.float32),
        'stride': 3,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: NWC, float32, stride 1, padding VALID, dilation 3
    input_dict = {
        'input': np.random.randn(2, 5, 4).astype(np.float32),
        'filters': np.random.randn(1, 4, 4).astype(np.float32),
        'stride': 1,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 3,
        'name': 'conv1d_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: NWC, float64, stride 2, padding SAME, dilation 1
    input_dict = {
        'input': np.random.randn(1, 30, 2).astype(np.float64),
        'filters': np.random.randn(4, 2, 2).astype(np.float64),
        'stride': 2,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: NWC, float32, stride 1, padding VALID, dilation 1
    input_dict = {
        'input': np.random.randn(5, 7, 8).astype(np.float32),
        'filters': np.random.randn(2, 8, 16).astype(np.float32),
        'stride': 1,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D batch_shape NWC, float32, stride 1, padding SAME, dilation 1
    input_dict = {
        'input': np.random.randn(2, 2, 10, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 5).astype(np.float32),
        'stride': 1,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Explicit values, float32, NWC, stride 1, padding VALID, dilation 1
    input_dict = {
        'input': np.array([[[-1.0, 2.0], [0.5, -1.5], [2.0, -3.0]]], dtype=np.float32),
        'filters': np.array([[[1.0, -1.0], [2.0, 0.0]]], dtype=np.float32),
        'stride': 1,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.conv1d_3"] = generate_conv1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.conv1d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv1d_3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.conv1d', generated_inputs['tf.nn.conv1d_3'], lib="tf", suffix=3)
