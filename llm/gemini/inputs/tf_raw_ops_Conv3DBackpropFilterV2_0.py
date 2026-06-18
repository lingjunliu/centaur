
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_Conv3DBackpropFilterV2_inputs():
    list_of_inputs = []

    # Input 1: NDHWC, float32, VALID padding, strides=1, dilations=1
    input_dict_1 = {
        "input": np.random.randn(2, 5, 5, 5, 3).astype(np.float32),
        "filter_sizes": np.array([3, 3, 3, 3, 4], dtype=np.int32),
        "out_backprop": np.random.randn(2, 3, 3, 3, 4).astype(np.float32),
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: NDHWC, float32, VALID padding, strides=2, dilations=1
    input_dict_2 = {
        "input": np.random.randn(1, 8, 8, 8, 1).astype(np.float32),
        "filter_sizes": np.array([2, 2, 2, 1, 2], dtype=np.int32),
        "out_backprop": np.random.randn(1, 4, 4, 4, 2).astype(np.float32),
        "strides": [1, 2, 2, 2, 1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: NDHWC, float32, SAME padding, strides=1, dilations=1
    input_dict_3 = {
        "input": np.random.randn(2, 5, 5, 5, 3).astype(np.float32),
        "filter_sizes": np.array([3, 3, 3, 3, 4], dtype=np.int32),
        "out_backprop": np.random.randn(2, 5, 5, 5, 4).astype(np.float32),
        "strides": [1, 1, 1, 1, 1],
        "padding": "SAME",
        "data_format": "NDHWC",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: NDHWC, float32, SAME padding, strides=2, dilations=1
    input_dict_4 = {
        "input": np.random.randn(1, 6, 6, 6, 2).astype(np.float32),
        "filter_sizes": np.array([3, 3, 3, 2, 2], dtype=np.int32),
        "out_backprop": np.random.randn(1, 3, 3, 3, 2).astype(np.float32),
        "strides": [1, 2, 2, 2, 1],
        "padding": "SAME",
        "data_format": "NDHWC",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: NDHWC, float64, VALID padding, strides=1, dilations=1
    input_dict_5 = {
        "input": np.random.randn(2, 4, 4, 4, 2).astype(np.float64),
        "filter_sizes": np.array([2, 2, 2, 2, 1], dtype=np.int32),
        "out_backprop": np.random.randn(2, 3, 3, 3, 1).astype(np.float64),
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: NDHWC, float16, VALID padding, strides=1, dilations=1
    input_dict_6 = {
        "input": np.random.randn(1, 3, 3, 3, 1).astype(np.float16),
        "filter_sizes": np.array([2, 2, 2, 1, 1], dtype=np.int32),
        "out_backprop": np.random.randn(1, 2, 2, 2, 1).astype(np.float16),
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: NCDHW, float32, VALID padding, strides=1, dilations=1
    input_dict_7 = {
        "input": np.random.randn(2, 3, 5, 5, 5).astype(np.float32),
        "filter_sizes": np.array([3, 3, 3, 3, 4], dtype=np.int32),
        "out_backprop": np.random.randn(2, 4, 3, 3, 3).astype(np.float32),
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NCDHW",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: NCDHW, float32, SAME padding, strides=2, dilations=1
    input_dict_8 = {
        "input": np.random.randn(1, 2, 4, 4, 4).astype(np.float32),
        "filter_sizes": np.array([2, 2, 2, 2, 3], dtype=np.int32),
        "out_backprop": np.random.randn(1, 3, 2, 2, 2).astype(np.float32),
        "strides": [1, 1, 2, 2, 2],
        "padding": "SAME",
        "data_format": "NCDHW",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: NDHWC, float32, VALID padding, strides=1, dilations=2
    input_dict_9 = {
        "input": np.random.randn(2, 7, 7, 7, 2).astype(np.float32),
        "filter_sizes": np.array([3, 3, 3, 2, 2], dtype=np.int32),
        "out_backprop": np.random.randn(2, 3, 3, 3, 2).astype(np.float32),
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "dilations": [1, 2, 2, 2, 1],
        "name": "conv3d_backprop_filter_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: NCDHW, float32, VALID padding, strides=1, dilations=2
    input_dict_10 = {
        "input": np.random.randn(2, 2, 7, 7, 7).astype(np.float32),
        "filter_sizes": np.array([3, 3, 3, 2, 2], dtype=np.int32),
        "out_backprop": np.random.randn(2, 2, 3, 3, 3).astype(np.float32),
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NCDHW",
        "dilations": [1, 1, 2, 2, 2],
        "name": "conv3d_backprop_filter_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.Conv3DBackpropFilterV2"] = tf_raw_ops_Conv3DBackpropFilterV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Conv3DBackpropFilterV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv3DBackpropFilterV2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Conv3DBackpropFilterV2', generated_inputs['tf.raw_ops.Conv3DBackpropFilterV2'], lib="tf", suffix=0)
