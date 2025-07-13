
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ExtractVolumePatches_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case with float32
    input1 = np.float32(np.random.rand(1, 3, 10, 10, 1))
    ksizes1 = [1, 1, 3, 3, 1]
    strides1 = [1, 1, 2, 2, 1]
    padding1 = "VALID"

    input_dict1 = {
        "input": tf.constant(input1, dtype=tf.float32),
        "ksizes": ksizes1,
        "strides": strides1,
        "padding": padding1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different ksizes and strides with int32
    input2 = np.int32(np.random.randint(0, 10, size=(2, 5, 8, 8, 3)))
    ksizes2 = [1, 2, 2, 2, 1]
    strides2 = [1, 2, 1, 1, 1]
    padding2 = "SAME"

    input_dict2 = {
        "input": tf.constant(input2, dtype=tf.int32),
        "ksizes": ksizes2,
        "strides": strides2,
        "padding": padding2,
        "name": "extract_patches_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different data type (uint8)
    input3 = np.uint8(np.random.randint(0, 256, size=(1, 4, 6, 6, 1)))
    ksizes3 = [1, 2, 3, 3, 1]
    strides3 = [1, 1, 1, 1, 1]
    padding3 = "VALID"

    input_dict3 = {
        "input": tf.constant(input3, dtype=tf.uint8),
        "ksizes": ksizes3,
        "strides": strides3,
        "padding": padding3,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger batch size
    input4 = np.float32(np.random.rand(4, 2, 5, 5, 2))
    ksizes4 = [1, 1, 2, 2, 1]
    strides4 = [1, 1, 1, 1, 1]
    padding4 = "SAME"

    input_dict4 = {
        "input": tf.constant(input4, dtype=tf.float32),
        "ksizes": ksizes4,
        "strides": strides4,
        "padding": padding4,
        "name": "extract_patches_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: int64 data type
    input5 = np.int64(np.random.randint(-100, 100, size=(1, 3, 7, 7, 1)))
    ksizes5 = [1, 1, 3, 3, 1]
    strides5 = [1, 1, 2, 2, 1]
    padding5 = "VALID"

    input_dict5 = {
        "input": tf.constant(input5, dtype=tf.int64),
        "ksizes": ksizes5,
        "strides": strides5,
        "padding": padding5,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different padding SAME with larger strides
    input6 = np.float32(np.random.rand(1, 3, 10, 10, 1))
    ksizes6 = [1, 1, 3, 3, 1]
    strides6 = [1, 1, 3, 3, 1]
    padding6 = "SAME"

    input_dict6 = {
        "input": tf.constant(input6, dtype=tf.float32),
        "ksizes": ksizes6,
        "strides": strides6,
        "padding": padding6,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Larger depth
    input7 = np.float32(np.random.rand(1, 3, 10, 10, 5))
    ksizes7 = [1, 1, 3, 3, 1]
    strides7 = [1, 1, 2, 2, 1]
    padding7 = "VALID"

    input_dict7 = {
        "input": tf.constant(input7, dtype=tf.float32),
        "ksizes": ksizes7,
        "strides": strides7,
        "padding": padding7,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

     # Input 8: bfloat16 data type
    input8 = np.float32(np.random.rand(1, 3, 7, 7, 1)).astype(np.float16)
    ksizes8 = [1, 1, 3, 3, 1]
    strides8 = [1, 1, 2, 2, 1]
    padding8 = "VALID"

    input_dict8 = {
        "input": tf.constant(input8, dtype=tf.bfloat16),
        "ksizes": ksizes8,
        "strides": strides8,
        "padding": padding8,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Large input with valid
    input9 = np.float32(np.random.rand(2, 4, 12, 12, 3))
    ksizes9 = [1, 2, 4, 4, 1]
    strides9 = [1, 2, 2, 2, 1]
    padding9 = "VALID"
    input_dict9 = {
        "input": tf.constant(input9, dtype=tf.float32),
        "ksizes": ksizes9,
        "strides": strides9,
        "padding": padding9,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: uint32
    input10 = np.uint32(np.random.randint(0, 100, size=(1, 3, 7, 7, 1)))
    ksizes10 = [1, 1, 3, 3, 1]
    strides10 = [1, 1, 2, 2, 1]
    padding10 = "VALID"

    input_dict10 = {
        "input": tf.constant(input10, dtype=tf.uint32),
        "ksizes": ksizes10,
        "strides": strides10,
        "padding": padding10,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ExtractVolumePatches"] = tf_raw_ops_ExtractVolumePatches_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ExtractVolumePatches' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ExtractVolumePatches'.")

check_valid('tf.raw_ops.ExtractVolumePatches', generated_inputs['tf.raw_ops.ExtractVolumePatches'], lib="tf", suffix=0)
