
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_avgpool3d_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    input1 = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    ksize1 = [1, 2, 2, 2, 1]
    strides1 = [1, 1, 1, 1, 1]
    padding1 = "VALID"
    data_format1 = "NDHWC"
    name1 = "avgpool1"

    input_dict1 = {
        "input": input1,
        "ksize": ksize1,
        "strides": strides1,
        "padding": padding1,
        "data_format": data_format1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: SAME padding
    input2 = np.random.rand(1, 7, 7, 7, 3).astype(np.float32)
    ksize2 = [1, 3, 3, 3, 1]
    strides2 = [1, 1, 1, 1, 1]
    padding2 = "SAME"
    data_format2 = "NDHWC"
    name2 = "avgpool2"

    input_dict2 = {
        "input": input2,
        "ksize": ksize2,
        "strides": strides2,
        "padding": padding2,
        "data_format": data_format2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Strides > 1
    input3 = np.random.rand(1, 10, 10, 10, 3).astype(np.float32)
    ksize3 = [1, 2, 2, 2, 1]
    strides3 = [1, 2, 2, 2, 1]
    padding3 = "VALID"
    data_format3 = "NDHWC"
    name3 = "avgpool3"

    input_dict3 = {
        "input": input3,
        "ksize": ksize3,
        "strides": strides3,
        "padding": padding3,
        "data_format": data_format3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different data format (NCDHW)
    input4 = np.random.rand(1, 3, 5, 5, 5).astype(np.float32)
    ksize4 = [1, 2, 2, 2, 1]
    strides4 = [1, 1, 1, 1, 1]
    padding4 = "VALID"
    data_format4 = "NCDHW"
    name4 = "avgpool4"

    input_dict4 = {
        "input": input4,
        "ksize": ksize4,
        "strides": strides4,
        "padding": padding4,
        "data_format": data_format4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: ksize and strides different sizes
    input5 = np.random.rand(1, 6, 6, 6, 3).astype(np.float32)
    ksize5 = [1, 2, 2, 2, 1]
    strides5 = [1, 3, 3, 3, 1]
    padding5 = "VALID"
    data_format5 = "NDHWC"
    name5 = "avgpool5"

    input_dict5 = {
        "input": input5,
        "ksize": ksize5,
        "strides": strides5,
        "padding": padding5,
        "data_format": data_format5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: ksize and strides same size
    input6 = np.random.rand(1, 8, 8, 8, 3).astype(np.float32)
    ksize6 = [1, 4, 4, 4, 1]
    strides6 = [1, 4, 4, 4, 1]
    padding6 = "VALID"
    data_format6 = "NDHWC"
    name6 = "avgpool6"

    input_dict6 = {
        "input": input6,
        "ksize": ksize6,
        "strides": strides6,
        "padding": padding6,
        "data_format": data_format6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: small input size with SAME padding
    input7 = np.random.rand(1, 2, 2, 2, 3).astype(np.float32)
    ksize7 = [1, 3, 3, 3, 1]
    strides7 = [1, 1, 1, 1, 1]
    padding7 = "SAME"
    data_format7 = "NDHWC"
    name7 = "avgpool7"

    input_dict7 = {
        "input": input7,
        "ksize": ksize7,
        "strides": strides7,
        "padding": padding7,
        "data_format": data_format7,
        "name": name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Larger kernel and stride
    input8 = np.random.rand(1, 16, 16, 16, 3).astype(np.float32)
    ksize8 = [1, 5, 5, 5, 1]
    strides8 = [1, 3, 3, 3, 1]
    padding8 = "VALID"
    data_format8 = "NDHWC"
    name8 = "avgpool8"

    input_dict8 = {
        "input": input8,
        "ksize": ksize8,
        "strides": strides8,
        "padding": padding8,
        "data_format": data_format8,
        "name": name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Larger input size
    input9 = np.random.rand(1, 32, 32, 32, 3).astype(np.float32)
    ksize9 = [1, 4, 4, 4, 1]
    strides9 = [1, 2, 2, 2, 1]
    padding9 = "VALID"
    data_format9 = "NDHWC"
    name9 = "avgpool9"

    input_dict9 = {
        "input": input9,
        "ksize": ksize9,
        "strides": strides9,
        "padding": padding9,
        "data_format": data_format9,
        "name": name9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Larger batch size
    input10 = np.random.rand(4, 8, 8, 8, 3).astype(np.float32)
    ksize10 = [1, 2, 2, 2, 1]
    strides10 = [1, 2, 2, 2, 1]
    padding10 = "VALID"
    data_format10 = "NDHWC"
    name10 = "avgpool10"

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
generated_inputs["tf.raw_ops.AvgPool3D"] = tf_raw_ops_avgpool3d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AvgPool3D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AvgPool3D'.")

check_valid('tf.raw_ops.AvgPool3D', generated_inputs['tf.raw_ops.AvgPool3D'], lib="tf", suffix=0)
