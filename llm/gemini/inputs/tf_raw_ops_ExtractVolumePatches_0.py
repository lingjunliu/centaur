
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_extract_volume_patches_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 3, 32, 32, 3).astype(np.float32)
    ksizes1 = [1, 1, 3, 3, 1]
    strides1 = [1, 1, 2, 2, 1]
    padding1 = "VALID"
    name1 = "extract_patches_1"

    input_dict1 = {
        "input": input1,
        "ksizes": ksizes1,
        "strides": strides1,
        "padding": padding1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(2, 5, 64, 64, 1).astype(np.float64)
    ksizes2 = [1, 3, 5, 5, 1]
    strides2 = [1, 2, 3, 3, 1]
    padding2 = "SAME"
    name2 = "extract_patches_2"

    input_dict2 = {
        "input": input2,
        "ksizes": ksizes2,
        "strides": strides2,
        "padding": padding2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.random.randint(0, 10, size=(1, 2, 16, 16, 2), dtype=np.int32)
    ksizes3 = [1, 1, 4, 4, 1]
    strides3 = [1, 1, 4, 4, 1]
    padding3 = "VALID"
    name3 = "extract_patches_3"

    input_dict3 = {
        "input": input3,
        "ksizes": ksizes3,
        "strides": strides3,
        "padding": padding3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.random.randint(0, 256, size=(4, 4, 8, 8, 1), dtype=np.uint8)
    ksizes4 = [1, 2, 2, 2, 1]
    strides4 = [1, 1, 1, 1, 1]
    padding4 = "SAME"
    name4 = "extract_patches_4"

    input_dict4 = {
        "input": input4,
        "ksizes": ksizes4,
        "strides": strides4,
        "padding": padding4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.random.randint(-100, 100, size=(1, 3, 12, 12, 3), dtype=np.int16)
    ksizes5 = [1, 1, 5, 5, 1]
    strides5 = [1, 1, 3, 3, 1]
    padding5 = "VALID"
    name5 = "extract_patches_5"

    input_dict5 = {
        "input": input5,
        "ksizes": ksizes5,
        "strides": strides5,
        "padding": padding5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.random.randint(-50, 50, size=(2, 2, 20, 20, 1), dtype=np.int8)
    ksizes6 = [1, 2, 7, 7, 1]
    strides6 = [1, 1, 5, 5, 1]
    padding6 = "SAME"
    name6 = "extract_patches_6"

    input_dict6 = {
        "input": input6,
        "ksizes": ksizes6,
        "strides": strides6,
        "padding": padding6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = np.random.randint(0, 1000, size=(1, 4, 32, 32, 4), dtype=np.int64)
    ksizes7 = [1, 1, 1, 1, 1]
    strides7 = [1, 1, 1, 1, 1]
    padding7 = "VALID"
    name7 = "extract_patches_7"

    input_dict7 = {
        "input": input7,
        "ksizes": ksizes7,
        "strides": strides7,
        "padding": padding7,
        "name": name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.random.rand(1, 3, 32, 32, 3).astype(np.float16)
    ksizes8 = [1, 1, 3, 3, 1]
    strides8 = [1, 1, 2, 2, 1]
    padding8 = "VALID"
    name8 = "extract_patches_8"

    input_dict8 = {
        "input": input8,
        "ksizes": ksizes8,
        "strides": strides8,
        "padding": padding8,
        "name": name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = np.random.rand(2, 5, 64, 64, 1).astype(np.float32)
    ksizes9 = [1, 3, 5, 5, 1]
    strides9 = [1, 2, 3, 3, 1]
    padding9 = "SAME"
    name9 = "extract_patches_9"

    input_dict9 = {
        "input": input9,
        "ksizes": ksizes9,
        "strides": strides9,
        "padding": padding9,
        "name": name9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = np.random.rand(2, 5, 64, 64, 1).astype(np.half)
    ksizes10 = [1, 3, 5, 5, 1]
    strides10 = [1, 2, 3, 3, 1]
    padding10 = "SAME"
    name10 = "extract_patches_10"

    input_dict10 = {
        "input": input10,
        "ksizes": ksizes10,
        "strides": strides10,
        "padding": padding10,
        "name": name10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ExtractVolumePatches"] = tf_raw_ops_extract_volume_patches_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ExtractVolumePatches' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ExtractVolumePatches'.")

check_valid('tf.raw_ops.ExtractVolumePatches', generated_inputs['tf.raw_ops.ExtractVolumePatches'], lib="tf", suffix=0)
