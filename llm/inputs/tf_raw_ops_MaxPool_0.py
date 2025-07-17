
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_maxpool_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    ksize1 = [1, 2, 2, 1]
    strides1 = [1, 1, 1, 1]
    padding1 = "VALID"

    input_dict = {
        "input": input1,
        "ksize": ksize1,
        "strides": strides1,
        "padding": padding1,
        "explicit_paddings": [],
        "data_format": "NHWC",
        "name": "maxpool1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input2 = np.random.rand(1, 10, 10, 1).astype(np.float32)
    ksize2 = [1, 3, 3, 1]
    strides2 = [1, 2, 2, 1]
    padding2 = "SAME"

    input_dict = {
        "input": input2,
        "ksize": ksize2,
        "strides": strides2,
        "padding": padding2,
        "explicit_paddings": [],
        "data_format": "NHWC",
        "name": "maxpool2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input3 = np.random.rand(2, 7, 7, 5).astype(np.float32)
    ksize3 = [1, 4, 4, 1]
    strides3 = [1, 3, 3, 1]
    padding3 = "VALID"

    input_dict = {
        "input": input3,
        "ksize": ksize3,
        "strides": strides3,
        "padding": padding3,
        "explicit_paddings": [],
        "data_format": "NHWC",
        "name": "maxpool3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input4 = np.random.rand(1, 3, 3, 1).astype(np.float32)
    ksize4 = [1, 2, 2, 1]
    strides4 = [1, 1, 1, 1]
    padding4 = "EXPLICIT"
    explicit_paddings4 = [0, 0, 0, 0, 0, 0, 0, 0]


    input_dict = {
        "input": input4,
        "ksize": ksize4,
        "strides": strides4,
        "padding": padding4,
        "explicit_paddings": explicit_paddings4,
        "data_format": "NHWC",
        "name": "maxpool4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input5 = np.random.rand(1, 3, 3, 1).astype(np.float32)
    ksize5 = [1, 2, 2, 1]
    strides5 = [1, 1, 1, 1]
    padding5 = "SAME"
    explicit_paddings5 = []

    input_dict = {
        "input": input5,
        "ksize": ksize5,
        "strides": strides5,
        "padding": padding5,
        "explicit_paddings": explicit_paddings5,
        "data_format": "NHWC",
        "name": "maxpool5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (NCHW)
    input6 = np.random.rand(1, 3, 5, 5).astype(np.float32)
    ksize6 = [1, 1, 2, 2]
    strides6 = [1, 1, 1, 1]
    padding6 = "VALID"

    input_dict = {
        "input": input6,
        "ksize": ksize6,
        "strides": strides6,
        "padding": padding6,
        "explicit_paddings": [],
        "data_format": "NCHW",
        "name": "maxpool6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (int32)
    input7 = np.random.randint(0, 10, size=(1, 5, 5, 3)).astype(np.int32)
    ksize7 = [1, 2, 2, 1]
    strides7 = [1, 1, 1, 1]
    padding7 = "VALID"

    input_dict = {
        "input": input7,
        "ksize": ksize7,
        "strides": strides7,
        "padding": padding7,
        "explicit_paddings": [],
        "data_format": "NHWC",
        "name": "maxpool7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (different batch size)
    input8 = np.random.rand(4, 5, 5, 3).astype(np.float32)
    ksize8 = [1, 2, 2, 1]
    strides8 = [1, 1, 1, 1]
    padding8 = "VALID"

    input_dict = {
        "input": input8,
        "ksize": ksize8,
        "strides": strides8,
        "padding": padding8,
        "explicit_paddings": [],
        "data_format": "NHWC",
        "name": "maxpool8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (different number of channels)
    input9 = np.random.rand(1, 5, 5, 7).astype(np.float32)
    ksize9 = [1, 2, 2, 1]
    strides9 = [1, 1, 1, 1]
    padding9 = "VALID"

    input_dict = {
        "input": input9,
        "ksize": ksize9,
        "strides": strides9,
        "padding": padding9,
        "explicit_paddings": [],
        "data_format": "NHWC",
        "name": "maxpool9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input10 = np.random.rand(1, 10, 10, 1).astype(np.float32)
    ksize10 = [1, 3, 3, 1]
    strides10 = [1, 2, 2, 1]
    padding10 = "SAME"

    input_dict = {
        "input": input10,
        "ksize": ksize10,
        "strides": strides10,
        "padding": padding10,
        "explicit_paddings": [],
        "data_format": "NHWC",
        "name": "maxpool10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MaxPool"] = tf_raw_ops_maxpool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MaxPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPool'.")

check_valid('tf.raw_ops.MaxPool', generated_inputs['tf.raw_ops.MaxPool'], lib="tf", suffix=0)
