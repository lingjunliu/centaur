
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_depth_to_space_inputs():
    list_of_inputs = []

    # Input 1: Simple NHWC example
    input1 = np.array([[[[1, 2, 3, 4]]]], dtype=np.float32)
    block_size1 = 2
    data_format1 = "NHWC"
    name1 = None

    input_dict1 = {
        "input": input1,
        "block_size": block_size1,
        "data_format": data_format1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Larger depth NHWC
    input2 = np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]], dtype=np.float32)
    block_size2 = 2
    data_format2 = "NHWC"
    name2 = "depth_to_space_example_2"

    input_dict2 = {
        "input": input2,
        "block_size": block_size2,
        "data_format": data_format2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: NHWC with larger input
    input3 = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]], dtype=np.float32)
    block_size3 = 2
    data_format3 = "NHWC"
    name3 = None

    input_dict3 = {
        "input": input3,
        "block_size": block_size3,
        "data_format": data_format3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: NCHW example
    input4 = np.array([[[[1, 2], [3, 4], [5, 6], [7, 8]]]], dtype=np.float32)
    block_size4 = 2
    data_format4 = "NCHW"
    name4 = "depth_to_space_example_4"

    input_dict4 = {
        "input": input4,
        "block_size": block_size4,
        "data_format": data_format4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different block size
    input5 = np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9]]]], dtype=np.float32)
    block_size5 = 3
    data_format5 = "NHWC"
    name5 = None

    input_dict5 = {
        "input": input5,
        "block_size": block_size5,
        "data_format": data_format5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Batch size > 1
    input6 = np.array([[[[1, 2, 3, 4]]], [[[5, 6, 7, 8]]]], dtype=np.float32)
    block_size6 = 2
    data_format6 = "NHWC"
    name6 = "depth_to_space_example_6"

    input_dict6 = {
        "input": input6,
        "block_size": block_size6,
        "data_format": data_format6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

     # Input 7: NCHW with larger input
    input7 = np.array([[[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]]]], dtype=np.float32)
    block_size7 = 2
    data_format7 = "NCHW"
    name7 = None

    input_dict7 = {
        "input": input7,
        "block_size": block_size7,
        "data_format": data_format7,
        "name": name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))


    # Input 8: float64 input
    input8 = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float64)
    block_size8 = 2
    data_format8 = "NHWC"
    name8 = None

    input_dict8 = {
        "input": input8,
        "block_size": block_size8,
        "data_format": data_format8,
        "name": name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: block size 4
    input9 = np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]]], dtype=np.float32)
    block_size9 = 4
    data_format9 = "NHWC"
    name9 = "depth_to_space_example_9"

    input_dict9 = {
        "input": input9,
        "block_size": block_size9,
        "data_format": data_format9,
        "name": name9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

     # Input 10: NCHW_VECT_C format. Reduce depth to be divisible by 4
    input10 = np.array([[[[[1, 2, 3, 4], [5, 6, 7, 8]]]]], dtype=np.int8)
    block_size10 = 2
    data_format10 = "NCHW_VECT_C"
    name10 = "depth_to_space_example_10"

    input_dict10 = {
        "input": input10,
        "block_size": block_size10,
        "data_format": data_format10,
        "name": name10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
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
