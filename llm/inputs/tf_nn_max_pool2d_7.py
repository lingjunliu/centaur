
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    ksize1 = [1, 1, 1, 1]
    strides1 = [1, 1, 1, 1]
    padding1 = 'VALID'
    data_format1 = 'NHWC'
    name1 = 'pool1'

    input_dict1 = {
        "input": input1,
        "ksize": ksize1,
        "strides": strides1,
        "padding": padding1,
        "data_format": data_format1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
    ksize2 = [1, 2, 2, 1]
    strides2 = [1, 2, 2, 1]
    padding2 = 'VALID'
    data_format2 = 'NHWC'
    name2 = 'pool2'

    input_dict2 = {
        "input": input2,
        "ksize": ksize2,
        "strides": strides2,
        "padding": padding2,
        "data_format": data_format2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    ksize3 = [1, 2, 2, 1]
    strides3 = [1, 1, 1, 1]
    padding3 = 'VALID'
    data_format3 = 'NHWC'
    name3 = 'pool3'

    input_dict3 = {
        "input": input3,
        "ksize": ksize3,
        "strides": strides3,
        "padding": padding3,
        "data_format": data_format3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    ksize4 = [1, 2, 2, 1]
    strides4 = [1, 1, 1, 1]
    padding4 = 'SAME'
    data_format4 = 'NHWC'
    name4 = 'pool4'

    input_dict4 = {
        "input": input4,
        "ksize": ksize4,
        "strides": strides4,
        "padding": padding4,
        "data_format": data_format4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.float32)
    ksize5 = [1, 2, 2, 1]
    strides5 = [1, 1, 1, 1]
    padding5 = 'SAME'
    data_format5 = 'NHWC'
    name5 = 'pool5'

    input_dict5 = {
        "input": input5,
        "ksize": ksize5,
        "strides": strides5,
        "padding": padding5,
        "data_format": data_format5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    ksize6 = [1, 2, 2, 1]
    strides6 = [1, 1, 1, 1]
    padding6 = 'VALID'
    data_format6 = 'NHWC'
    name6 = 'pool6'

    input_dict6 = {
        "input": input6,
        "ksize": ksize6,
        "strides": strides6,
        "padding": padding6,
        "data_format": data_format6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    ksize7 = [1, 2, 2, 1]
    strides7 = [1, 1, 1, 1]
    padding7 = 'SAME'
    data_format7 = 'NHWC'
    name7 = 'pool7'

    input_dict7 = {
        "input": input7,
        "ksize": ksize7,
        "strides": strides7,
        "padding": padding7,
        "data_format": data_format7,
        "name": name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8
    input8 = np.array([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]], dtype=np.float32)
    ksize8 = [1, 2, 2, 1]
    strides8 = [1, 1, 1, 1]
    padding8 = 'VALID'
    data_format8 = 'NHWC'
    name8 = 'pool8'

    input_dict8 = {
        "input": input8,
        "ksize": ksize8,
        "strides": strides8,
        "padding": padding8,
        "data_format": data_format8,
        "name": name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9
    input9 = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    ksize9 = [1, 2, 2, 1]
    strides9 = [1, 1, 1, 1]
    padding9 = "VALID"
    data_format9 = 'NHWC'
    name9 = 'pool9'

    input_dict9 = {
        "input": input9,
        "ksize": ksize9,
        "strides": strides9,
        "padding": padding9,
        "data_format": data_format9,
        "name": name9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10
    input10 = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    ksize10 = [1, 2, 2, 1]
    strides10 = [1, 1, 1, 1]
    padding10 = "SAME"
    data_format10 = 'NHWC'
    name10 = 'pool10'

    input_dict10 = {
        "input": input10,
        "ksize": ksize10,
        "strides": strides10,
        "padding": padding10,
        "data_format": data_format10,
        "name": name10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
     # Input 11 - different ksize and strides
    input11 = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], dtype=np.float32)
    ksize11 = [1, 2, 2, 1]
    strides11 = [1, 2, 2, 1]
    padding11 = 'VALID'
    data_format11 = 'NHWC'
    name11 = 'pool11'

    input_dict11 = {
        "input": input11,
        "ksize": ksize11,
        "strides": strides11,
        "padding": padding11,
        "data_format": data_format11,
        "name": name11
    }
    list_of_inputs.append(copy.deepcopy(input_dict11))

    # Input 12 - Explicit Padding
    input12 = np.array([[[[1., 2., 3., 4.],
                     [5., 6., 7., 8.],
                     [9., 10., 11., 12.]]]], dtype=np.float32)
    ksize12 = [1, 2, 2, 1]
    strides12 = [1, 2, 2, 1]
    padding12 = [[0, 0], [1, 1], [1, 1], [0, 0]]
    data_format12 = 'NHWC'
    name12 = 'pool12'

    input_dict12 = {
        "input": input12,
        "ksize": ksize12,
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
