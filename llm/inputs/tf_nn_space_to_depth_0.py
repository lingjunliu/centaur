
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_space_to_depth_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1], [2]], [[3], [4]]]]).astype(np.float32)
    block_size = np.int32(2)
    data_format = "NHWC"
    name = "space_to_depth_example_1"
    input_dict = {"input": input_tensor, "block_size": block_size, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).astype(np.float32)
    block_size = np.int32(2)
    data_format = "NHWC"
    name = "space_to_depth_example_2"
    input_dict = {"input": input_tensor, "block_size": block_size, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[[1], [2]], [[5], [6]]], [[[3], [4]], [[7], [8]]], [[[9], [10]], [[13], [14]]], [[[11], [12]], [[15], [16]]]]]).astype(np.float32)
    input_tensor = np.reshape(input_tensor, (1,4,4,1))
    block_size = np.int32(2)
    data_format = "NHWC"
    name = "space_to_depth_example_3"
    input_dict = {"input": input_tensor, "block_size": block_size, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, different block size
    input_tensor = np.array([[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]]]]).astype(np.float32)
    input_tensor = np.reshape(input_tensor, (1,3,3,2))
    block_size = np.int32(3)
    data_format = "NHWC"
    name = "space_to_depth_example_4"
    input_dict = {"input": input_tensor, "block_size": block_size, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, NCHW
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32)
    input_tensor = np.reshape(input_tensor, (1,2,2,2))
    input_tensor = np.transpose(input_tensor, (0,3,1,2))
    block_size = np.int32(2)
    data_format = "NCHW"
    name = "space_to_depth_example_5"
    input_dict = {"input": input_tensor, "block_size": block_size, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, multi batch
    input_tensor = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]).astype(np.float32)
    block_size = np.int32(2)
    data_format = "NHWC"
    name = "space_to_depth_example_6"
    input_dict = {"input": input_tensor, "block_size": block_size, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, larger input
    input_tensor = np.random.rand(1, 8, 8, 3).astype(np.float32)
    block_size = np.int32(2)
    data_format = "NHWC"
    name = "space_to_depth_example_7"
    input_dict = {"input": input_tensor, "block_size": block_size, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, different data type
    input_tensor = np.array([[[[1], [2]], [[3], [4]]]]).astype(np.int32)
    block_size = np.int32(2)
    data_format = "NHWC"
    name = "space_to_depth_example_8"
    input_dict = {"input": input_tensor, "block_size": block_size, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, block_size = 4
    input_tensor = np.arange(1, 17).reshape((1, 4, 4, 1)).astype(np.float32)
    block_size = np.int32(4)
    data_format = "NHWC"
    name = "space_to_depth_example_9"
    input_dict = {"input": input_tensor, "block_size": block_size, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, block_size = 2, different channels
    input_tensor = np.arange(1, 49).reshape((1, 4, 4, 3)).astype(np.float32)
    block_size = np.int32(2)
    data_format = "NHWC"
    name = "space_to_depth_example_10"
    input_dict = {"input": input_tensor, "block_size": block_size, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.space_to_depth"] = tf_nn_space_to_depth_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.space_to_depth' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.space_to_depth'.")

check_valid('tf.nn.space_to_depth', generated_inputs['tf.nn.space_to_depth'], lib="tf", suffix=0)
