
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32)
    ksize1 = 2
    strides1 = [1, 1, 1, 1]
    padding1 = [[0, 0], [0, 0], [0, 0], [0, 0]]
    data_format1 = 'NHWC'
    name1 = None

    input_dict1 = {
        "input": input1,
        "ksize": np.int32(ksize1),
        "strides": strides1,
        "padding": padding1,
        "data_format": data_format1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).astype(np.float32)
    ksize2 = 1
    strides2 = [2, 2]
    padding2 = "VALID"
    data_format2 = 'NHWC'
    name2 = None

    input_dict2 = {
        "input": input2,
        "ksize": np.int32(ksize2),
        "strides": strides2,
        "padding": padding2,
        "data_format": data_format2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.array([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]]).astype(np.float32)
    ksize3 = 2
    strides3 = [1, 2]
    padding3 = "SAME"
    data_format3 = 'NHWC'
    name3 = None

    input_dict3 = {
        "input": input3,
        "ksize": np.int32(ksize3),
        "strides": strides3,
        "padding": padding3,
        "data_format": data_format3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    ksize4 = 3
    strides4 = [1, 1, 1, 1]
    padding4 = [[0, 0], [1, 1], [1, 1], [0, 0]]
    data_format4 = 'NHWC'
    name4 = None

    input_dict4 = {
        "input": input4,
        "ksize": np.int32(ksize4),
        "strides": strides4,
        "padding": padding4,
        "data_format": data_format4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.random.rand(1, 10, 10, 1).astype(np.float32)
    ksize5 = 4
    strides5 = [4, 4]
    padding5 = 'VALID'
    data_format5 = 'NHWC'
    name5 = None

    input_dict5 = {
        "input": input5,
        "ksize": np.int32(ksize5),
        "strides": strides5,
        "padding": padding5,
        "data_format": data_format5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.random.rand(2, 8, 8, 3).astype(np.float32)
    ksize6 = 2
    strides6 = [2, 2, 2, 2]
    padding6 = 'SAME'
    data_format6 = 'NHWC'
    name6 = None

    input_dict6 = {
        "input": input6,
        "ksize": np.int32(ksize6),
        "strides": strides6,
        "padding": padding6,
        "data_format": data_format6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = np.random.rand(1, 4, 4, 1).astype(np.float32)
    ksize7 = 2
    strides7 = [1, 1]
    padding7 = [[0, 0], [0, 0], [0, 0], [0, 0]]
    data_format7 = 'NHWC'
    name7 = None

    input_dict7 = {
        "input": input7,
        "ksize": np.int32(ksize7),
        "strides": strides7,
        "padding": padding7,
        "data_format": data_format7,
        "name": name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.random.rand(1, 7, 7, 3).astype(np.float32)
    ksize8 = 3
    strides8 = [2, 2]
    padding8 = [[0, 0], [1, 0], [0, 1], [0, 0]]
    data_format8 = 'NHWC'
    name8 = None

    input_dict8 = {
        "input": input8,
        "ksize": np.int32(ksize8),
        "strides": strides8,
        "padding": padding8,
        "data_format": data_format8,
        "name": name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = np.random.rand(4, 4, 4, 4).astype(np.float32)
    ksize9 = 1
    strides9 = [1, 1, 1, 1]
    padding9 = 'VALID'
    data_format9 = 'NHWC'
    name9 = None

    input_dict9 = {
        "input": input9,
        "ksize": np.int32(ksize9),
        "strides": strides9,
        "padding": padding9,
        "data_format": data_format9,
        "name": name9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = np.random.rand(2, 5, 5, 2).astype(np.float32)
    ksize10 = 5
    strides10 = [1, 1]
    padding10 = 'SAME'
    data_format10 = 'NHWC'
    name10 = None

    input_dict10 = {
        "input": input10,
        "ksize": np.int32(ksize10),
        "strides": strides10,
        "padding": padding10,
        "data_format": data_format10,
        "name": name10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11 - explicit padding with different values
    input11 = np.random.rand(1, 3, 3, 1).astype(np.float32)
    ksize11 = 2
    strides11 = [1, 1]
    padding11 = [[0, 0], [0, 1], [1, 0], [0, 0]]
    data_format11 = 'NHWC'
    name11 = None
    
    input_dict11 = {
        "input": input11,
        "ksize": np.int32(ksize11),
        "strides": strides11,
        "padding": padding11,
        "data_format": data_format11,
        "name": name11
    }
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    # Input 12 - integer ksize
    input12 = np.random.rand(1, 4, 4, 1).astype(np.float32)
    ksize12 = 2
    strides12 = [1, 1]
    padding12 = "VALID"
    data_format12 = 'NHWC'
    name12 = None

    input_dict12 = {
        "input": input12,
        "ksize": np.int32(ksize12),
        "strides": strides12,
        "padding": padding12,
        "data_format": data_format12,
        "name": name12
    }
    list_of_inputs.append(copy.deepcopy(input_dict12))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.max_pool2d_7"] = tf_nn_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.max_pool2d_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool2d_7'.")

check_valid('tf.nn.max_pool2d', generated_inputs['tf.nn.max_pool2d_7'], lib="tf", suffix=7)
