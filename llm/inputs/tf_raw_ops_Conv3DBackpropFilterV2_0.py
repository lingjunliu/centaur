
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv3DBackpropFilterV2_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filter_sizes1 = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop1 = np.random.rand(1, 3, 3, 3, 2).astype(np.float32)
    strides1 = [1, 1, 1, 1, 1]
    padding1 = "VALID"
    data_format1 = "NDHWC"
    dilations1 = [1, 1, 1, 1, 1]
    name1 = "conv3d_backprop_filter_v2_1"

    input_dict1 = {
        "input": input1,
        "filter_sizes": filter_sizes1,
        "out_backprop": out_backprop1,
        "strides": strides1,
        "padding": padding1,
        "data_format": data_format1,
        "dilations": dilations1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(2, 10, 10, 10, 1).astype(np.float32)
    filter_sizes2 = np.array([5, 5, 5, 1, 4], dtype=np.int32)
    out_backprop2 = np.random.rand(2, 6, 6, 6, 4).astype(np.float32)
    strides2 = [1, 1, 1, 1, 1]
    padding2 = "VALID"
    data_format2 = "NDHWC"
    dilations2 = [1, 1, 1, 1, 1]
    name2 = "conv3d_backprop_filter_v2_2"

    input_dict2 = {
        "input": input2,
        "filter_sizes": filter_sizes2,
        "out_backprop": out_backprop2,
        "strides": strides2,
        "padding": padding2,
        "data_format": data_format2,
        "dilations": dilations2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.random.rand(1, 7, 7, 7, 3).astype(np.float32)
    filter_sizes3 = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop3 = np.random.rand(1, 7, 7, 7, 2).astype(np.float32)
    strides3 = [1, 1, 1, 1, 1]
    padding3 = "SAME"
    data_format3 = "NDHWC"
    dilations3 = [1, 1, 1, 1, 1]
    name3 = "conv3d_backprop_filter_v2_3"

    input_dict3 = {
        "input": input3,
        "filter_sizes": filter_sizes3,
        "out_backprop": out_backprop3,
        "strides": strides3,
        "padding": padding3,
        "data_format": data_format3,
        "dilations": dilations3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.random.rand(1, 7, 7, 7, 3).astype(np.float32)
    filter_sizes4 = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop4 = np.random.rand(1, 3, 3, 3, 2).astype(np.float32)
    strides4 = [1, 2, 2, 2, 1]
    padding4 = "VALID"
    data_format4 = "NDHWC"
    dilations4 = [1, 1, 1, 1, 1]
    name4 = "conv3d_backprop_filter_v2_4"

    input_dict4 = {
        "input": input4,
        "filter_sizes": filter_sizes4,
        "out_backprop": out_backprop4,
        "strides": strides4,
        "padding": padding4,
        "data_format": data_format4,
        "dilations": dilations4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

     # Input 5
    input5 = np.random.rand(1, 7, 7, 7, 3).astype(np.float32)
    filter_sizes5 = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop5 = np.random.rand(1, 7, 7, 7, 2).astype(np.float32)
    strides5 = [1, 1, 1, 1, 1]
    padding5 = "SAME"
    data_format5 = "NDHWC"
    dilations5 = [1, 1, 1, 1, 1]
    name5 = "conv3d_backprop_filter_v2_5"

    input_dict5 = {
        "input": input5,
        "filter_sizes": filter_sizes5,
        "out_backprop": out_backprop5,
        "strides": strides5,
        "padding": padding5,
        "data_format": data_format5,
        "dilations": dilations5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6 (Corrected to align input and filter depth in NCDHW)
    input6 = np.random.rand(2, 3, 5, 5, 5).astype(np.float32)
    filter_sizes6 = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop6 = np.random.rand(2, 3, 3, 3, 2).astype(np.float32)
    strides6 = [1, 1, 1, 1, 1]
    padding6 = "VALID"
    data_format6 = "NCDHW"
    dilations6 = [1, 1, 1, 1, 1]
    name6 = "conv3d_backprop_filter_v2_6"

    input_dict6 = {
        "input": input6,
        "filter_sizes": filter_sizes6,
        "out_backprop": out_backprop6,
        "strides": strides6,
        "padding": padding6,
        "data_format": data_format6,
        "dilations": dilations6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filter_sizes7 = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop7 = np.random.rand(1, 1, 1, 1, 2).astype(np.float32)
    strides7 = [1, 3, 3, 3, 1]
    padding7 = "VALID"
    data_format7 = "NDHWC"
    dilations7 = [1, 1, 1, 1, 1]
    name7 = "conv3d_backprop_filter_v2_7"

    input_dict7 = {
        "input": input7,
        "filter_sizes": filter_sizes7,
        "out_backprop": out_backprop7,
        "strides": strides7,
        "padding": padding7,
        "data_format": data_format7,
        "dilations": dilations7,
        "name": name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filter_sizes8 = np.array([3, 3, 3, 3, 2], dtype=np.int32)
    out_backprop8 = np.random.rand(1, 5, 5, 5, 2).astype(np.float32)
    strides8 = [1, 1, 1, 1, 1]
    padding8 = "SAME"
    data_format8 = "NDHWC"
    dilations8 = [1, 2, 2, 2, 1]
    name8 = "conv3d_backprop_filter_v2_8"

    input_dict8 = {
        "input": input8,
        "filter_sizes": filter_sizes8,
        "out_backprop": out_backprop8,
        "strides": strides8,
        "padding": padding8,
        "data_format": data_format8,
        "dilations": dilations8,
        "name": name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 10 (Corrected shape for NCDHW)
    input10 = np.random.rand(1, 3, 5, 5, 5).astype(np.float32)
    filter_sizes10 = np.array([3, 3, 3, 3, 2], dtype=np.int32)  # Filter size appropriate for NCDHW
    out_backprop10 = np.random.rand(1, 3, 5, 5, 2).astype(np.float32)
    strides10 = [1, 1, 1, 1, 1]
    padding10 = "SAME"
    data_format10 = "NCDHW"
    dilations10 = [1, 1, 1, 1, 1]
    name10 = "conv3d_backprop_filter_v2_10"

    input_dict10 = {
        "input": input10,
        "filter_sizes": filter_sizes10,
        "out_backprop": out_backprop10,
        "strides": strides10,
        "padding": padding10,
        "data_format": data_format10,
        "dilations": dilations10,
        "name": name10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Conv3DBackpropFilterV2"] = tf_raw_ops_Conv3DBackpropFilterV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Conv3DBackpropFilterV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv3DBackpropFilterV2'.")

check_valid('tf.raw_ops.Conv3DBackpropFilterV2', generated_inputs['tf.raw_ops.Conv3DBackpropFilterV2'], lib="tf", suffix=0)
