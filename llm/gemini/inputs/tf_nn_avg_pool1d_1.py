
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 5, 3).astype(np.float32)
    ksize1 = 2
    strides1 = 1
    padding1 = 'VALID'
    data_format1 = 'NWC'
    name1 = None

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
    input2 = np.random.rand(2, 10, 1).astype(np.float32)
    ksize2 = 3
    strides2 = 2
    padding2 = 'SAME'
    data_format2 = 'NWC'
    name2 = 'avg_pool'

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
    input3 = np.random.rand(1, 7, 5).astype(np.float32)
    ksize3 = 1
    strides3 = 1
    padding3 = 'VALID'
    data_format3 = 'NWC'
    name3 = None

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
    input4 = np.random.rand(3, 8, 2).astype(np.float32)
    ksize4 = 4
    strides4 = 3
    padding4 = 'SAME'
    data_format4 = 'NWC'
    name4 = 'avg_pool_2'

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
    input5 = np.random.rand(1, 12, 4).astype(np.float32)
    ksize5 = 5
    strides5 = 4
    padding5 = 'VALID'
    data_format5 = 'NWC'
    name5 = None

    input_dict5 = {
        "input": input5,
        "ksize": ksize5,
        "strides": strides5,
        "padding": padding5,
        "data_format": data_format5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: NCW format
    input6 = np.random.rand(2, 3, 6).astype(np.float32)
    ksize6 = 2
    strides6 = 1
    padding6 = 'VALID'
    data_format6 = 'NCW'
    name6 = None

    input_dict6 = {
        "input": input6,
        "ksize": ksize6,
        "strides": strides6,
        "padding": padding6,
        "data_format": data_format6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: NCW format, SAME padding
    input7 = np.random.rand(1, 4, 8).astype(np.float32)
    ksize7 = 3
    strides7 = 2
    padding7 = 'SAME'
    data_format7 = 'NCW'
    name7 = 'avg_pool_ncw'

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
    input8 = np.random.rand(1, 3, 1).astype(np.float32)
    ksize8 = 1
    strides8 = 1
    padding8 = 'VALID'
    data_format8 = 'NWC'
    name8 = None

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
    input9 = np.random.rand(4, 15, 10).astype(np.float32)
    ksize9 = 6
    strides9 = 5
    padding9 = 'SAME'
    data_format9 = 'NWC'
    name9 = 'avg_pool_9'

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
    input10 = np.random.rand(1, 4, 1).astype(np.float32)
    ksize10 = 4
    strides10 = 1
    padding10 = 'VALID'
    data_format10 = 'NWC'
    name10 = None

    input_dict10 = {
        "input": input10,
        "ksize": ksize10,
        "strides": strides10,
        "padding": padding10,
        "data_format": data_format10,
        "name": name10
    }
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
