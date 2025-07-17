
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPoolV2_inputs():
    list_of_inputs = []

    # Input 1: Basic NHWC, VALID padding
    input1 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    ksize1 = np.array([1, 2, 2, 1], dtype=np.int32)
    strides1 = np.array([1, 1, 1, 1], dtype=np.int32)
    padding1 = "VALID"
    data_format1 = "NHWC"

    input_dict1 = {
        "input": input1,
        "ksize": ksize1,
        "strides": strides1,
        "padding": padding1,
        "data_format": data_format1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Basic NHWC, SAME padding
    input2 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    ksize2 = np.array([1, 2, 2, 1], dtype=np.int32)
    strides2 = np.array([1, 1, 1, 1], dtype=np.int32)
    padding2 = "SAME"
    data_format2 = "NHWC"

    input_dict2 = {
        "input": input2,
        "ksize": ksize2,
        "strides": strides2,
        "padding": padding2,
        "data_format": data_format2,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different strides
    input3 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    ksize3 = np.array([1, 2, 2, 1], dtype=np.int32)
    strides3 = np.array([1, 2, 2, 1], dtype=np.int32)
    padding3 = "VALID"
    data_format3 = "NHWC"

    input_dict3 = {
        "input": input3,
        "ksize": ksize3,
        "strides": strides3,
        "padding": padding3,
        "data_format": data_format3,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

     # Input 4: Different ksize
    input4 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    ksize4 = np.array([1, 3, 3, 1], dtype=np.int32)
    strides4 = np.array([1, 1, 1, 1], dtype=np.int32)
    padding4 = "VALID"
    data_format4 = "NHWC"

    input_dict4 = {
        "input": input4,
        "ksize": ksize4,
        "strides": strides4,
        "padding": padding4,
        "data_format": data_format4,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Integer Input
    input5 = np.random.randint(0, 10, size=(1, 5, 5, 3)).astype(np.int32)
    ksize5 = np.array([1, 2, 2, 1], dtype=np.int32)
    strides5 = np.array([1, 1, 1, 1], dtype=np.int32)
    padding5 = "VALID"
    data_format5 = "NHWC"

    input_dict5 = {
        "input": input5,
        "ksize": ksize5,
        "strides": strides5,
        "padding": padding5,
        "data_format": data_format5,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: ksize and strides as 1
    input6 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    ksize6 = np.array([1, 1, 1, 1], dtype=np.int32)
    strides6 = np.array([1, 1, 1, 1], dtype=np.int32)
    padding6 = "VALID"
    data_format6 = "NHWC"

    input_dict6 = {
        "input": input6,
        "ksize": ksize6,
        "strides": strides6,
        "padding": padding6,
        "data_format": data_format6,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Larger Batch Size
    input7 = np.random.rand(4, 5, 5, 3).astype(np.float32)
    ksize7 = np.array([1, 2, 2, 1], dtype=np.int32)
    strides7 = np.array([1, 1, 1, 1], dtype=np.int32)
    padding7 = "VALID"
    data_format7 = "NHWC"

    input_dict7 = {
        "input": input7,
        "ksize": ksize7,
        "strides": strides7,
        "padding": padding7,
        "data_format": data_format7,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Different data type int16
    input8 = np.random.randint(0, 10, size=(1, 5, 5, 3)).astype(np.int16)
    ksize8 = np.array([1, 2, 2, 1], dtype=np.int32)
    strides8 = np.array([1, 1, 1, 1], dtype=np.int32)
    padding8 = "VALID"
    data_format8 = "NHWC"

    input_dict8 = {
        "input": input8,
        "ksize": ksize8,
        "strides": strides8,
        "padding": padding8,
        "data_format": data_format8,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: NCHW_VECT_C, VALID padding
    input9 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    ksize9 = np.array([1, 2, 2, 1], dtype=np.int32)
    strides9 = np.array([1, 1, 1, 1], dtype=np.int32)
    padding9 = "VALID"
    data_format9 = "NCHW_VECT_C"

    input_dict9 = {
        "input": input9,
        "ksize": ksize9,
        "strides": strides9,
        "padding": padding9,
        "data_format": data_format9,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Different strides with NHWC_VECT_C
    input10 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    ksize10 = np.array([1, 2, 2, 1], dtype=np.int32)
    strides10 = np.array([1, 2, 2, 1], dtype=np.int32)
    padding10 = "VALID"
    data_format10 = "NCHW_VECT_C"

    input_dict10 = {
        "input": input10,
        "ksize": ksize10,
        "strides": strides10,
        "padding": padding10,
        "data_format": data_format10,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MaxPoolV2"] = tf_raw_ops_MaxPoolV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MaxPoolV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolV2'.")

check_valid('tf.raw_ops.MaxPoolV2', generated_inputs['tf.raw_ops.MaxPoolV2'], lib="tf", suffix=0)
