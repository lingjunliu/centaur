
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
    ksize1 = [1, 2, 2, 1]
    strides1 = 1
    padding1 = [[0, 0], [0, 0], [0, 0], [0, 0]]
    data_format1 = 'NHWC'
    name1 = 'max_pool_1'
    input_dict1 = {'input': input1, 'ksize': ksize1, 'strides': strides1, 'padding': padding1, 'data_format': data_format1, 'name': name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).astype(np.float32)
    ksize2 = [1, 2, 3, 1]
    strides2 = 2
    padding2 = [[0, 0], [1, 0], [0, 1], [0, 0]]
    data_format2 = 'NHWC'
    name2 = 'max_pool_2'
    input_dict2 = {'input': input2, 'ksize': ksize2, 'strides': strides2, 'padding': padding2, 'data_format': data_format2, 'name': name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    ksize3 = [1, 3, 3, 1]
    strides3 = 3
    padding3 = [[0, 0], [0, 0], [0, 0], [0, 0]]
    data_format3 = 'NHWC'
    name3 = 'max_pool_3'
    input_dict3 = {'input': input3, 'ksize': ksize3, 'strides': strides3, 'padding': padding3, 'data_format': data_format3, 'name': name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.random.rand(2, 10, 10, 1).astype(np.float32)
    ksize4 = [1, 5, 5, 1]
    strides4 = 4
    padding4 = [[0, 0], [2, 2], [2, 2], [0, 0]]
    data_format4 = 'NHWC'
    name4 = 'max_pool_4'
    input_dict4 = {'input': input4, 'ksize': ksize4, 'strides': strides4, 'padding': padding4, 'data_format': data_format4, 'name': name4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.random.rand(1, 20, 20, 3).astype(np.float32)
    ksize5 = [1, 7, 7, 1]
    strides5 = 5
    padding5 = [[0, 0], [3, 3], [3, 3], [0, 0]]
    data_format5 = 'NHWC'
    name5 = 'max_pool_5'
    input_dict5 = {'input': input5, 'ksize': ksize5, 'strides': strides5, 'padding': padding5, 'data_format': data_format5, 'name': name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.random.rand(4, 8, 8, 2).astype(np.float32)
    ksize6 = [1, 4, 4, 1]
    strides6 = 2
    padding6 = [[0, 0], [1, 1], [1, 1], [0, 0]]
    data_format6 = 'NHWC'
    name6 = 'max_pool_6'
    input_dict6 = {'input': input6, 'ksize': ksize6, 'strides': strides6, 'padding': padding6, 'data_format': data_format6, 'name': name6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = np.random.rand(1, 12, 12, 1).astype(np.float32)
    ksize7 = [1, 6, 6, 1]
    strides7 = 3
    padding7 = [[0, 0], [2, 2], [2, 2], [0, 0]]
    data_format7 = 'NHWC'
    name7 = 'max_pool_7'
    input_dict7 = {'input': input7, 'ksize': ksize7, 'strides': strides7, 'padding': padding7, 'data_format': data_format7, 'name': name7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.random.rand(2, 16, 16, 3).astype(np.float32)
    ksize8 = [1, 8, 8, 1]
    strides8 = 4
    padding8 = [[0, 0], [3, 3], [3, 3], [0, 0]]
    data_format8 = 'NHWC'
    name8 = 'max_pool_8'
    input_dict8 = {'input': input8, 'ksize': ksize8, 'strides': strides8, 'padding': padding8, 'data_format': data_format8, 'name': name8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = np.random.rand(1, 28, 28, 1).astype(np.float32)
    ksize9 = [1, 14, 14, 1]
    strides9 = 7
    padding9 = [[0, 0], [6, 6], [6, 6], [0, 0]]
    data_format9 = 'NHWC'
    name9 = 'max_pool_9'
    input_dict9 = {'input': input9, 'ksize': ksize9, 'strides': strides9, 'padding': padding9, 'data_format': data_format9, 'name': name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = np.random.rand(4, 32, 32, 3).astype(np.float32)
    ksize10 = [1, 16, 16, 1]
    strides10 = 8
    padding10 = [[0, 0], [7, 7], [7, 7], [0, 0]]
    data_format10 = 'NHWC'
    name10 = 'max_pool_10'
    input_dict10 = {'input': input10, 'ksize': ksize10, 'strides': strides10, 'padding': padding10, 'data_format': data_format10, 'name': name10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
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
