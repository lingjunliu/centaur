
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 5, 1).astype(np.float32)
    ksize1 = 2
    strides1 = 1
    padding1 = 'VALID'
    data_format1 = 'NWC'
    name1 = 'avgpool1'
    input_dict1 = {'input': tf.convert_to_tensor(input1), 'ksize': int(ksize1), 'strides': int(strides1), 'padding': padding1, 'data_format': data_format1, 'name': name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(2, 10, 3).astype(np.float32)
    ksize2 = 3
    strides2 = 2
    padding2 = 'SAME'
    data_format2 = 'NWC'
    name2 = None
    input_dict2 = {'input': tf.convert_to_tensor(input2), 'ksize': int(ksize2), 'strides': int(strides2), 'padding': padding2, 'data_format': data_format2, 'name': name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.random.rand(4, 7, 2).astype(np.float32)
    ksize3 = 1
    strides3 = 1
    padding3 = 'VALID'
    data_format3 = 'NWC'
    name3 = 'avgpool3'
    input_dict3 = {'input': tf.convert_to_tensor(input3), 'ksize': int(ksize3), 'strides': int(strides3), 'padding': padding3, 'data_format': data_format3, 'name': name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.random.rand(1, 15, 4).astype(np.float32)
    ksize4 = 4
    strides4 = 3
    padding4 = 'SAME'
    data_format4 = 'NWC'
    name4 = None
    input_dict4 = {'input': tf.convert_to_tensor(input4), 'ksize': int(ksize4), 'strides': int(strides4), 'padding': padding4, 'data_format': data_format4, 'name': name4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.random.rand(2, 8, 1).astype(np.float32)
    ksize5 = 2
    strides5 = 1
    padding5 = 'VALID'
    data_format5 = 'NCW'
    name5 = 'avgpool5'
    input_dict5 = {'input': tf.transpose(tf.convert_to_tensor(input5), perm=[0, 2, 1]), 'ksize': int(ksize5), 'strides': int(strides5), 'padding': padding5, 'data_format': data_format5, 'name': name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.random.rand(1, 20, 5).astype(np.float32)
    ksize6 = 5
    strides6 = 4
    padding6 = 'SAME'
    data_format6 = 'NWC'
    name6 = None
    input_dict6 = {'input': tf.convert_to_tensor(input6), 'ksize': int(ksize6), 'strides': int(strides6), 'padding': padding6, 'data_format': data_format6, 'name': name6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = np.random.rand(3, 12, 2).astype(np.float32)
    ksize7 = 3
    strides7 = 2
    padding7 = 'VALID'
    data_format7 = 'NWC'
    name7 = 'avgpool7'
    input_dict7 = {'input': tf.convert_to_tensor(input7), 'ksize': int(ksize7), 'strides': int(strides7), 'padding': padding7, 'data_format': data_format7, 'name': name7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.random.rand(1, 6, 3).astype(np.float32)
    ksize8 = 1
    strides8 = 1
    padding8 = 'SAME'
    data_format8 = 'NWC'
    name8 = None
    input_dict8 = {'input': tf.convert_to_tensor(input8), 'ksize': int(ksize8), 'strides': int(strides8), 'padding': padding8, 'data_format': data_format8, 'name': name8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = np.random.rand(2, 18, 4).astype(np.float32)
    ksize9 = 6
    strides9 = 5
    padding9 = 'VALID'
    data_format9 = 'NWC'
    name9 = 'avgpool9'
    input_dict9 = {'input': tf.convert_to_tensor(input9), 'ksize': int(ksize9), 'strides': int(strides9), 'padding': padding9, 'data_format': data_format9, 'name': name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = np.random.rand(1, 9, 1).astype(np.float32)
    ksize10 = 3
    strides10 = 3
    padding10 = 'SAME'
    data_format10 = 'NWC'
    name10 = None
    input_dict10 = {'input': tf.convert_to_tensor(input10), 'ksize': int(ksize10), 'strides': int(strides10), 'padding': padding10, 'data_format': data_format10, 'name': name10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.avg_pool1d_1"] = tf_nn_avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.avg_pool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.avg_pool1d_1'.")

check_valid('tf.nn.avg_pool1d', generated_inputs['tf.nn.avg_pool1d_1'], lib="tf", suffix=1)
