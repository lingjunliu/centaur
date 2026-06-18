
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_bias_add_inputs():
    list_of_inputs = []

    # Input 1: float32, NHWC format, 2D array (N, C)
    value = np.random.randn(2, 3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NHWC',
        'name': 'bias_add_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, NHWC format, 2D array (N, C)
    value = np.random.randint(-10, 10, size=(3, 5)).astype(np.int32)
    bias = np.random.randint(-5, 5, size=(5,)).astype(np.int32)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NHWC',
        'name': 'bias_add_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, NHWC format, 3D array (N, H, C)
    value = np.random.randn(2, 4, 3).astype(np.float64)
    bias = np.random.randn(3).astype(np.float64)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NHWC',
        'name': 'bias_add_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, NCHW format, 3D array (N, C, H)
    value = np.random.randn(2, 3, 4).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NCHW',
        'name': 'bias_add_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, NHWC format, 4D array (N, H, W, C)
    value = np.random.randn(2, 5, 5, 4).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NHWC',
        'name': 'bias_add_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, NCHW format, 4D array (N, C, H, W)
    value = np.random.randn(2, 4, 5, 5).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NCHW',
        'name': 'bias_add_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64, NHWC format, 2D array (N, C)
    value = np.random.randint(-100, 100, size=(10, 2)).astype(np.int64)
    bias = np.random.randint(-50, 50, size=(2,)).astype(np.int64)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NHWC',
        'name': 'bias_add_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, NCHW format, 4D array (N, C, H, W)
    value = np.random.randn(1, 3, 8, 8).astype(np.float64)
    bias = np.random.randn(3).astype(np.float64)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NCHW',
        'name': 'bias_add_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, NHWC format, 4D array with larger channels
    value = np.random.randn(4, 3, 3, 16).astype(np.float32)
    bias = np.random.randn(16).astype(np.float32)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NHWC',
        'name': 'bias_add_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, NCHW format, 4D array with larger channels
    value = np.random.randn(4, 16, 3, 3).astype(np.float32)
    bias = np.random.randn(16).astype(np.float32)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NCHW',
        'name': 'bias_add_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.bias_add"] = tf_nn_bias_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.bias_add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.bias_add'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.bias_add', generated_inputs['tf.nn.bias_add'], lib="tf", suffix=0)
