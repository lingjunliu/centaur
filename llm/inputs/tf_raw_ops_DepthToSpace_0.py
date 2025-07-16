
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_depth_to_space_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1, 2, 3, 4]]]], dtype=np.float32)
    block_size = 2
    data_format = 'NHWC'
    name = None

    input_dict = {
        "input": input_tensor,
        "block_size": block_size,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]], dtype=np.float32)
    block_size = 2
    data_format = 'NHWC'
    name = "test_op"

    input_dict = {
        "input": input_tensor,
        "block_size": block_size,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]], dtype=np.float32)
    block_size = 2
    data_format = 'NHWC'
    name = None

    input_dict = {
        "input": input_tensor,
        "block_size": block_size,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[[[1, 2, 3, 4]]]]], dtype=np.float32)
    block_size = 2
    data_format = 'NCHW'
    name = None

    input_dict = {
        "input": input_tensor,
        "block_size": block_size,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]], dtype=np.float32)
    block_size = 2
    data_format = 'NCHW'
    name = None

    input_dict = {
        "input": input_tensor,
        "block_size": block_size,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]]], dtype=np.float32)
    block_size = 2
    data_format = 'NCHW'
    name = None

    input_dict = {
        "input": input_tensor,
        "block_size": block_size,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input_tensor = np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]]], dtype=np.float32)
    block_size = 4
    data_format = 'NHWC'
    name = None

    input_dict = {
        "input": input_tensor,
        "block_size": block_size,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]]], dtype=np.float32)
    block_size = 4
    data_format = 'NCHW'
    name = None

    input_dict = {
        "input": input_tensor,
        "block_size": block_size,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[[1, 2, 3, 4]]]], dtype=np.int32)
    block_size = 2
    data_format = 'NHWC'
    name = None

    input_dict = {
        "input": input_tensor,
        "block_size": block_size,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]], dtype=np.int32)
    block_size = 2
    data_format = 'NHWC'
    name = None

    input_dict = {
        "input": input_tensor,
        "block_size": block_size,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: block_size = 3, depth divisible by 9
    input_tensor = np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9]]]], dtype=np.float32)
    block_size = 3
    data_format = 'NHWC'
    name = None
    
    input_dict = {
        "input": input_tensor,
        "block_size": block_size,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DepthToSpace"] = tf_raw_ops_depth_to_space_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DepthToSpace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DepthToSpace'.")

check_valid('tf.raw_ops.DepthToSpace', generated_inputs['tf.raw_ops.DepthToSpace'], lib="tf", suffix=0)
