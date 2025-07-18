
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic NHWC, no padding, 2x2 stride
    input_dict_1 = {
        'input': np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4, 1),
        'ksize': 2,
        'strides': [1, 2, 2, 1],
        'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: NHWC, no padding, 1x1 stride (overlapping)
    # Changed from NCHW to NHWC to support CPU execution.
    input_dict_2 = {
        'input': np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4, 1),
        'ksize': 2,
        'strides': [1, 1, 1, 1],
        'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: NHWC with symmetric explicit padding
    input_dict_3 = {
        'input': np.arange(1, 10, dtype=np.float32).reshape(1, 3, 3, 1),
        'ksize': 2,
        'strides': [1, 2, 2, 1],
        'padding': [[0, 0], [1, 1], [1, 1], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Overlapping pool with larger input
    input_dict_4 = {
        'input': np.arange(1, 26, dtype=np.float32).reshape(1, 5, 5, 1),
        'ksize': 2,
        'strides': [1, 1, 1, 1],
        'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Multiple channels (NHWC)
    input_dict_5 = {
        'input': np.arange(1, 33, dtype=np.float32).reshape(1, 4, 4, 2),
        'ksize': 2,
        'strides': [1, 2, 2, 1],
        'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Multiple batches (NHWC)
    input_dict_6 = {
        'input': np.arange(1, 33, dtype=np.float32).reshape(2, 4, 4, 1),
        'ksize': 2,
        'strides': [1, 2, 2, 1],
        'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: NHWC with asymmetric explicit padding
    # Changed from NCHW to NHWC to support CPU execution.
    input_dict_7 = {
        'input': np.arange(1, 10, dtype=np.float32).reshape(1, 3, 3, 1),
        'ksize': 2,
        'strides': [1, 1, 1, 1],
        'padding': [[0, 0], [1, 0], [0, 1], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Larger ksize
    input_dict_8 = {
        'input': np.arange(1, 26, dtype=np.float32).reshape(1, 5, 5, 1),
        'ksize': 3,
        'strides': [1, 1, 1, 1],
        'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Non-square input with padding
    input_dict_9 = {
        'input': np.arange(1, 19, dtype=np.float32).reshape(1, 3, 6, 1),
        'ksize': 2,
        'strides': [1, 2, 2, 1],
        'padding': [[0, 0], [1, 0], [0, 1], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: float64 data type
    input_dict_10 = {
        'input': np.arange(1, 17, dtype=np.float64).reshape(1, 4, 4, 1),
        'ksize': 2,
        'strides': [1, 2, 2, 1],
        'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Larger padding, ksize=3 to ensure padding <= ksize_dim
    input_dict_11 = {
        'input': np.arange(1, 10, dtype=np.float32).reshape(1, 3, 3, 1),
        'ksize': 3,
        'strides': [1, 1, 1, 1],
        'padding': [[0, 0], [1, 2], [2, 1], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool_11'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Multiple batches and channels with non-square HxW
    # Changed from NCHW to NHWC to support CPU execution.
    input_dict_12 = {
        'input': np.arange(1, 73, dtype=np.float32).reshape(2, 4, 3, 3),
        'ksize': 2,
        'strides': [1, 2, 2, 1],
        'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool_12'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.nn.max_pool2d_7"] = tf_nn_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.max_pool2d_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool2d_7'.")

check_valid('tf.nn.max_pool2d', generated_inputs['tf.nn.max_pool2d_7'], lib="tf", suffix=7)
