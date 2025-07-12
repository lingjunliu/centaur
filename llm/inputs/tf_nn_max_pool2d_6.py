
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case with SAME padding
    input1 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    ksize1 = [1, 2, 2, 1]
    strides1 = 2
    padding1 = "SAME"
    data_format1 = 'NHWC'
    name1 = 'pool1'
    input_dict1 = {'input': input1, 'ksize': ksize1, 'strides': strides1, 'padding': padding1, 'data_format': data_format1, 'name': name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Basic valid case with VALID padding
    input2 = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
    ksize2 = [1, 2, 2, 1]
    strides2 = 1
    padding2 = "VALID"
    data_format2 = 'NHWC'
    name2 = 'pool2'
    input_dict2 = {'input': input2, 'ksize': ksize2, 'strides': strides2, 'padding': padding2, 'data_format': data_format2, 'name': name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different ksize and strides, SAME padding
    input3 = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]], dtype=np.float32)
    ksize3 = [1, 3, 3, 1]
    strides3 = 2
    padding3 = "SAME"
    data_format3 = 'NHWC'
    name3 = 'pool3'
    input_dict3 = {'input': input3, 'ksize': ksize3, 'strides': strides3, 'padding': padding3, 'data_format': data_format3, 'name': name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different ksize and strides, VALID padding
    input4 = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]], dtype=np.float32)
    ksize4 = [1, 2, 2, 1]
    strides4 = 3
    padding4 = "VALID"
    data_format4 = 'NHWC'
    name4 = 'pool4'
    input_dict4 = {'input': input4, 'ksize': ksize4, 'strides': strides4, 'padding': padding4, 'data_format': data_format4, 'name': name4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Multiple channels
    input5 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float32)
    ksize5 = [1, 2, 2, 1]
    strides5 = 2
    padding5 = "SAME"
    data_format5 = 'NHWC'
    name5 = 'pool5'
    input_dict5 = {'input': input5, 'ksize': ksize5, 'strides': strides5, 'padding': padding5, 'data_format': data_format5, 'name': name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Removed Explicit padding, NHWC

    # Input 7: Different data format NCHW (removed to avoid CPU error)

    # Input 8: Removed Explicit padding, NCHW

    # Input 9: Single integer for ksize and stride
    input9 = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]], dtype=np.float32)
    ksize9 = [2]
    strides9 = 2
    padding9 = "SAME"
    data_format9 = 'NHWC'
    name9 = 'pool9'
    input_dict9 = {'input': input9, 'ksize': ksize9, 'strides': strides9, 'padding': padding9, 'data_format': data_format9, 'name': name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: ksize as a single integer, strides as list
    input10 = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]], dtype=np.float32)
    ksize10 = [2]
    strides10 = 2
    padding10 = "VALID"
    data_format10 = 'NHWC'
    name10 = 'pool10'
    input_dict10 = {'input': input10, 'ksize': ksize10, 'strides': strides10, 'padding': padding10, 'data_format': data_format10, 'name': name10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: One example with negative values
    input11 = np.array([[[[-1, -2], [-3, -4]], [[-5, -6], [-7, -8]]]], dtype=np.float32)
    ksize11 = [1, 2, 2, 1]
    strides11 = 2
    padding11 = "SAME"
    data_format11 = 'NHWC'
    name11 = 'pool11'
    input_dict11 = {'input': input11, 'ksize': ksize11, 'strides': strides11, 'padding': padding11, 'data_format': data_format11, 'name': name11}
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.max_pool2d_6"] = tf_nn_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.max_pool2d_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool2d_6'.")

check_valid('tf.nn.max_pool2d', generated_inputs['tf.nn.max_pool2d_6'], lib="tf", suffix=6)
