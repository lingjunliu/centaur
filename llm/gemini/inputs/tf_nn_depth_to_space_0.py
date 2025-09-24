
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_depth_to_space_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1, 2, 3, 4]]]], dtype=np.float32)
    block_size = 2
    data_format = 'NHWC'
    name = 'depth_to_space_example_1'
    input_dict = {'input': input_tensor, 'block_size': block_size, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]], dtype=np.float32)
    block_size = 2
    data_format = 'NHWC'
    name = 'depth_to_space_example_2'
    input_dict = {'input': input_tensor, 'block_size': block_size, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]], dtype=np.float32)
    block_size = 2
    data_format = 'NHWC'
    name = 'depth_to_space_example_3'
    input_dict = {'input': input_tensor, 'block_size': block_size, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, batch size 2
    input_tensor = np.array([[[[1, 2, 3, 4]]], [[[5, 6, 7, 8]]]], dtype=np.float32)
    block_size = 2
    data_format = 'NHWC'
    name = 'depth_to_space_example_5'
    input_dict = {'input': input_tensor, 'block_size': block_size, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, different block size
    input_tensor = np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9]]]], dtype=np.float32)
    block_size = 3
    data_format = 'NHWC'
    name = 'depth_to_space_example_6'
    input_dict = {'input': input_tensor, 'block_size': block_size, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, Larger input
    input_tensor = np.random.rand(1, 4, 4, 16).astype(np.float32)
    block_size = 2
    data_format = 'NHWC'
    name = 'depth_to_space_example_8'
    input_dict = {'input': input_tensor, 'block_size': block_size, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, blocksize = 4
    input_tensor = np.random.rand(1, 1, 1, 16).astype(np.float32)
    block_size = 4
    data_format = 'NHWC'
    name = 'depth_to_space_example_9'
    input_dict = {'input': input_tensor, 'block_size': block_size, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]]], dtype=np.float32)
    block_size = 2
    data_format = 'NHWC'
    name = 'depth_to_space_example_11'
    input_dict = {'input': input_tensor, 'block_size': block_size, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]]]], dtype=np.float32)
    block_size = 4
    data_format = 'NHWC'
    name = 'depth_to_space_example_12'
    input_dict = {'input': input_tensor, 'block_size': block_size, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(2, 2, 2, 4).astype(np.float32)
    block_size = 2
    data_format = 'NHWC'
    name = 'depth_to_space_example_13'
    input_dict = {'input': input_tensor, 'block_size': block_size, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.depth_to_space"] = tf_nn_depth_to_space_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.depth_to_space' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.depth_to_space'.")

check_valid('tf.nn.depth_to_space', generated_inputs['tf.nn.depth_to_space'], lib="tf", suffix=0)
