
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv3D_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case with NDHWC
    input1 = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filter1 = np.random.rand(3, 3, 3, 3, 2).astype(np.float32)
    strides1 = [1, 1, 1, 1, 1]
    padding1 = "VALID"
    data_format1 = "NDHWC"
    dilations1 = [1, 1, 1, 1, 1]
    name1 = "conv3d_basic"

    input_dict1 = {
        "input": tf.constant(input1).numpy(),
        "filter": tf.constant(filter1).numpy(),
        "strides": strides1,
        "padding": padding1,
        "data_format": data_format1,
        "dilations": dilations1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Basic valid case with NCDHW
    input2 = np.random.rand(1, 3, 5, 5, 5).astype(np.float32)
    filter2 = np.random.rand(3, 3, 3, 3, 2).astype(np.float32)
    strides2 = [1, 1, 1, 1, 1]
    padding2 = "SAME"
    data_format2 = "NCDHW"
    dilations2 = [1, 1, 1, 1, 1]
    name2 = "conv3d_ncdhw"

    input_dict2 = {
        "input": tf.constant(input2).numpy(),
        "filter": tf.constant(filter2).numpy(),
        "strides": strides2,
        "padding": padding2,
        "data_format": data_format2,
        "dilations": dilations2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different strides
    input3 = np.random.rand(1, 10, 10, 10, 3).astype(np.float32)
    filter3 = np.random.rand(3, 3, 3, 3, 2).astype(np.float32)
    strides3 = [1, 2, 2, 2, 1]
    padding3 = "VALID"
    data_format3 = "NDHWC"
    dilations3 = [1, 1, 1, 1, 1]
    name3 = "conv3d_strides"

    input_dict3 = {
        "input": tf.constant(input3).numpy(),
        "filter": tf.constant(filter3).numpy(),
        "strides": strides3,
        "padding": padding3,
        "data_format": data_format3,
        "dilations": dilations3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different dilations
    input4 = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filter4 = np.random.rand(3, 3, 3, 3, 2).astype(np.float32)
    strides4 = [1, 1, 1, 1, 1]
    padding4 = "VALID"
    data_format4 = "NDHWC"
    dilations4 = [1, 2, 2, 2, 1]
    name4 = "conv3d_dilations"

    input_dict4 = {
        "input": tf.constant(input4).numpy(),
        "filter": tf.constant(filter4).numpy(),
        "strides": strides4,
        "padding": padding4,
        "data_format": data_format4,
        "dilations": dilations4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Half type
    input5 = np.random.rand(1, 5, 5, 5, 3).astype(np.float16)
    filter5 = np.random.rand(3, 3, 3, 3, 2).astype(np.float16)
    strides5 = [1, 1, 1, 1, 1]
    padding5 = "VALID"
    data_format5 = "NDHWC"
    dilations5 = [1, 1, 1, 1, 1]
    name5 = "conv3d_half"

    input_dict5 = {
        "input": tf.constant(input5).numpy(),
        "filter": tf.constant(filter5).numpy(),
        "strides": strides5,
        "padding": padding5,
        "data_format": data_format5,
        "dilations": dilations5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: float64 type
    input7 = np.random.rand(1, 5, 5, 5, 3).astype(np.float64)
    filter7 = np.random.rand(3, 3, 3, 3, 2).astype(np.float64)
    strides7 = [1, 1, 1, 1, 1]
    padding7 = "VALID"
    data_format7 = "NDHWC"
    dilations7 = [1, 1, 1, 1, 1]
    name7 = "conv3d_float64"

    input_dict7 = {
        "input": tf.constant(input7).numpy(),
        "filter": tf.constant(filter7).numpy(),
        "strides": strides7,
        "padding": padding7,
        "data_format": data_format7,
        "dilations": dilations7,
        "name": name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Batch size > 1
    input8 = np.random.rand(2, 5, 5, 5, 3).astype(np.float32)
    filter8 = np.random.rand(3, 3, 3, 3, 2).astype(np.float32)
    strides8 = [1, 1, 1, 1, 1]
    padding8 = "VALID"
    data_format8 = "NDHWC"
    dilations8 = [1, 1, 1, 1, 1]
    name8 = "conv3d_batch"

    input_dict8 = {
        "input": tf.constant(input8).numpy(),
        "filter": tf.constant(filter8).numpy(),
        "strides": strides8,
        "padding": padding8,
        "data_format": data_format8,
        "dilations": dilations8,
        "name": name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: in_channels != out_channels
    input9 = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filter9 = np.random.rand(3, 3, 3, 3, 5).astype(np.float32)
    strides9 = [1, 1, 1, 1, 1]
    padding9 = "VALID"
    data_format9 = "NDHWC"
    dilations9 = [1, 1, 1, 1, 1]
    name9 = "conv3d_channels"

    input_dict9 = {
        "input": tf.constant(input9).numpy(),
        "filter": tf.constant(filter9).numpy(),
        "strides": strides9,
        "padding": padding9,
        "data_format": data_format9,
        "dilations": dilations9,
        "name": name9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

   # Input 10: Different filter size
    input10 = np.random.rand(1, 10, 10, 10, 3).astype(np.float32)
    filter10 = np.random.rand(5, 5, 5, 3, 2).astype(np.float32)
    strides10 = [1, 2, 2, 2, 1]
    padding10 = "VALID"
    data_format10 = "NDHWC"
    dilations10 = [1, 1, 1, 1, 1]
    name10 = "conv3d_filter_size"

    input_dict10 = {
        "input": tf.constant(input10).numpy(),
        "filter": tf.constant(filter10).numpy(),
        "strides": strides10,
        "padding": padding10,
        "data_format": data_format10,
        "dilations": dilations10,
        "name": name10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))



    # Remove the bfloat16 example, as it is causing issues with dtype conversion
    # Remove the bfloat16 example
    list_of_inputs.pop(5)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Conv3D"] = tf_raw_ops_Conv3D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Conv3D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv3D'.")

check_valid('tf.raw_ops.Conv3D', generated_inputs['tf.raw_ops.Conv3D'], lib="tf", suffix=0)
