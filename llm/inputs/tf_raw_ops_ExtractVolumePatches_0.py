
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ExtractVolumePatches_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 3, 10, 10, 3).astype(np.float32)
    ksizes = [1, 1, 3, 3, 1]
    strides = [1, 1, 2, 2, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "ksizes": ksizes,
        "strides": strides,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 5, 8, 8, 1).astype(np.float64)
    ksizes = [1, 3, 3, 3, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "SAME"

    input_dict = {
        "input": input_tensor,
        "ksizes": ksizes,
        "strides": strides,
        "padding": padding,
        "name": "extract_patches"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.randint(0, 10, size=(1, 4, 6, 6, 2), dtype=np.int32)
    ksizes = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "ksizes": ksizes,
        "strides": strides,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.randint(0, 256, size=(1, 2, 4, 4, 1), dtype=np.uint8)
    ksizes = [1, 1, 2, 2, 1]
    strides = [1, 1, 2, 2, 1]
    padding = "SAME"

    input_dict = {
        "input": input_tensor,
        "ksizes": ksizes,
        "strides": strides,
        "padding": padding,
        "name": "extract_patches"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.randint(-100, 100, size=(2, 3, 5, 5, 3), dtype=np.int16)
    ksizes = [1, 1, 3, 3, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "ksizes": ksizes,
        "strides": strides,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.randint(-50, 50, size=(1, 2, 3, 3, 1), dtype=np.int8)
    ksizes = [1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"

    input_dict = {
        "input": input_tensor,
        "ksizes": ksizes,
        "strides": strides,
        "padding": padding,
        "name": "extract_patches"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.randint(-1000, 1000, size=(1, 3, 7, 7, 2), dtype=np.int64)
    ksizes = [1, 2, 3, 3, 1]
    strides = [1, 1, 2, 2, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "ksizes": ksizes,
        "strides": strides,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    input_tensor = np.random.rand(1, 3, 10, 10, 3).astype(np.float32)
    ksizes = [1, 3, 5, 5, 1]
    strides = [1, 1, 2, 2, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "ksizes": ksizes,
        "strides": strides,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(2, 5, 8, 8, 1).astype(np.float64)
    ksizes = [1, 1, 3, 3, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"

    input_dict = {
        "input": input_tensor,
        "ksizes": ksizes,
        "strides": strides,
        "padding": padding,
        "name": "extract_patches"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.randint(0, 10, size=(1, 4, 6, 6, 2), dtype=np.int32)
    ksizes = [1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "ksizes": ksizes,
        "strides": strides,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
