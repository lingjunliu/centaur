
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv1d_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    input_tensor = np.random.rand(1, 10, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "stride": 1,
        "padding": "VALID",
        "data_format": "NWC",
        "dilations": [1],
        "name": "conv1d_basic"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different stride
    input_tensor = np.random.rand(1, 10, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "stride": 2,
        "padding": "SAME",
        "data_format": "NWC",
        "dilations": [1],
        "name": "conv1d_stride"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different data format
    input_tensor = np.random.rand(1, 3, 10).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "stride": 1,
        "padding": "VALID",
        "data_format": "NCW",
        "dilations": [1],
        "name": "conv1d_dataformat"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different dilation
    input_tensor = np.random.rand(1, 10, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "stride": 1,
        "padding": "VALID",
        "data_format": "NWC",
        "dilations": [2],
        "name": "conv1d_dilation"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5: Batch size > 1
    input_tensor = np.random.rand(2, 10, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "stride": 1,
        "padding": "SAME",
        "data_format": "NWC",
        "dilations": [1],
        "name": "conv1d_batch"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float16
    input_tensor = np.random.rand(1, 10, 3).astype(np.float16)
    filters_tensor = np.random.rand(3, 3, 5).astype(np.float16)
    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "stride": 1,
        "padding": "VALID",
        "data_format": "NWC",
        "dilations": [1],
        "name": "conv1d_float16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64
    input_tensor = np.random.rand(1, 10, 3).astype(np.float64)
    filters_tensor = np.random.rand(3, 3, 5).astype(np.float64)
    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "stride": 1,
        "padding": "SAME",
        "data_format": "NWC",
        "dilations": [1],
        "name": "conv1d_float64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: stride as list
    input_tensor = np.random.rand(1, 10, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "stride": 1,
        "padding": "VALID",
        "data_format": "NWC",
        "dilations": [1],
        "name": "conv1d_stride_list"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: dilations as list length 3
    input_tensor = np.random.rand(1, 10, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "stride": 1,
        "padding": "VALID",
        "data_format": "NWC",
        "dilations": [1, 1, 1],
        "name": "conv1d_dilations_list"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Input shape (1, 5, 1)
    input_tensor = np.random.rand(1, 5, 1).astype(np.float32)
    filters_tensor = np.random.rand(3, 1, 2).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "filters": filters_tensor,
        "stride": 1,
        "padding": "VALID",
        "data_format": "NWC",
        "dilations": [1],
        "name": "conv1d_shape"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.conv1d"] = tf_nn_conv1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.conv1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv1d'.")

check_valid('tf.nn.conv1d', generated_inputs['tf.nn.conv1d'], lib="tf", suffix=0)
