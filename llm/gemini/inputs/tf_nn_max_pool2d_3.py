
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []

    # Case 1: Standard pool, NHWC, VALID padding
    input_dict_1 = {
        'input': np.random.randn(1, 4, 4, 1).astype(np.float32),
        'ksize': 2,
        'strides': [2, 2],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: SAME padding, NHWC, stride 1
    input_dict_2 = {
        'input': np.random.randn(2, 8, 8, 3).astype(np.float32),
        'ksize': 3,
        'strides': [1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: NCHW format, VALID padding
    input_dict_3 = {
        'input': np.random.randn(1, 3, 10, 10).astype(np.float32),
        'ksize': 2,
        'strides': [2, 2],
        'padding': 'VALID',
        'data_format': 'NCHW',
        'name': 'pool_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Large ksize and strides, NHWC, VALID
    input_dict_4 = {
        'input': np.random.randn(4, 16, 16, 64).astype(np.float32),
        'ksize': 4,
        'strides': [4, 4],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: 1x1 stride, SAME padding
    input_dict_5 = {
        'input': np.random.randn(1, 14, 14, 1).astype(np.float32),
        'ksize': 2,
        'strides': [1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: ksize 1, stride 1 (effectively a no-op but valid)
    input_dict_6 = {
        'input': np.random.randn(3, 5, 5, 2).astype(np.float32),
        'ksize': 1,
        'strides': [1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: NCHW format, larger input, SAME padding
    input_dict_7 = {
        'input': np.random.randn(2, 4, 32, 32).astype(np.float32),
        'ksize': 2,
        'strides': [1, 1],
        'padding': 'SAME',
        'data_format': 'NCHW',
        'name': 'pool_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Large strides, matching ksize
    input_dict_8 = {
        'input': np.random.randn(1, 25, 25, 1).astype(np.float32),
        'ksize': 5,
        'strides': [5, 5],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: 3-channel input, typical image size, NHWC
    input_dict_9 = {
        'input': np.random.randn(2, 112, 112, 3).astype(np.float32),
        'ksize': 3,
        'strides': [2, 2],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: 3-channel input, typical image size, NCHW
    input_dict_10 = {
        'input': np.random.randn(2, 3, 112, 112).astype(np.float32),
        'ksize': 3,
        'strides': [2, 2],
        'padding': 'SAME',
        'data_format': 'NCHW',
        'name': 'pool_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.nn.max_pool2d_3"] = tf_nn_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.max_pool2d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool2d_3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.max_pool2d', generated_inputs['tf.nn.max_pool2d_3'], lib="tf", suffix=3)
