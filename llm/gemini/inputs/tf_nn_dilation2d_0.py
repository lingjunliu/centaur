
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_dilation2d_inputs():
    list_of_inputs = []

    # Input 1: Float32, SAME padding, stride 1, dilation 1
    input_dict = {
        'input': np.random.randn(1, 5, 5, 1).astype(np.float32),
        'filters': np.random.randn(3, 3, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64, VALID padding, stride 2, dilation 1
    input_dict = {
        'input': np.random.randn(2, 6, 6, 3).astype(np.float64),
        'filters': np.random.randn(2, 2, 3).astype(np.float64),
        'strides': [1, 2, 2, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Int32, SAME padding, stride 1, dilation 2
    input_dict = {
        'input': np.random.randint(-10, 10, size=(1, 5, 5, 2)).astype(np.int32),
        'filters': np.random.randint(-5, 5, size=(2, 2, 2)).astype(np.int32),
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 2, 2, 1],
        'name': "dilation_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Uint8, VALID padding, stride 1, dilation 1 (positive values only)
    input_dict = {
        'input': np.random.randint(0, 255, size=(1, 4, 4, 1)).astype(np.uint8),
        'filters': np.random.randint(0, 10, size=(2, 2, 1)).astype(np.uint8),
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float32, negative values, SAME padding, stride 1
    input_dict = {
        'input': (np.ones((1, 3, 3, 1), dtype=np.float32) * -5.0),
        'filters': np.zeros((2, 2, 1), dtype=np.float32),
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float32, VALID padding, stride 1, dilation 1
    input_dict = {
        'input': np.random.randn(2, 8, 8, 4).astype(np.float32),
        'filters': np.random.randn(3, 3, 4).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Int32, SAME padding, stride 2, dilation 1
    input_dict = {
        'input': np.random.randint(-50, 50, size=(1, 10, 10, 2)).astype(np.int32),
        'filters': np.random.randint(-5, 5, size=(3, 3, 2)).astype(np.int32),
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Int32, VALID padding, stride 1, dilation 2
    input_dict = {
        'input': np.random.randint(0, 1000, size=(1, 7, 7, 3)).astype(np.int32),
        'filters': np.random.randint(0, 50, size=(2, 2, 3)).astype(np.int32),
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 2, 2, 1],
        'name': "dilation_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float32, larger spatial dimensions, SAME padding, stride 3
    input_dict = {
        'input': np.random.randn(1, 15, 15, 1).astype(np.float32),
        'filters': np.random.randn(3, 3, 1).astype(np.float32),
        'strides': [1, 3, 3, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float32, asymmetric strides, VALID padding
    input_dict = {
        'input': np.random.randn(1, 6, 8, 2).astype(np.float32),
        'filters': np.random.randn(2, 3, 2).astype(np.float32),
        'strides': [1, 2, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.dilation2d"] = tf_nn_dilation2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.dilation2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.dilation2d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.dilation2d', generated_inputs['tf.nn.dilation2d'], lib="tf", suffix=0)
