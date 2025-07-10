
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_extract_volume_patches_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 3, 10, 10, 5).astype(np.float32)
    ksizes = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    name = "extract_patches_1"
    input_dict = {"input": input_tensor, "ksizes": ksizes, "strides": strides, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 5, 8, 8, 3).astype(np.float32)
    ksizes = [1, 3, 3, 3, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "SAME"
    name = "extract_patches_2"
    input_dict = {"input": input_tensor, "ksizes": ksizes, "strides": strides, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.randint(0, 10, size=(1, 4, 12, 12, 7)).astype(np.int32)
    ksizes = [1, 4, 4, 4, 1]
    strides = [1, 3, 3, 3, 1]
    padding = "VALID"
    name = "extract_patches_3"
    input_dict = {"input": input_tensor, "ksizes": ksizes, "strides": strides, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(3, 2, 6, 6, 2).astype(np.float64)
    ksizes = [1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    name = "extract_patches_4"
    input_dict = {"input": input_tensor, "ksizes": ksizes, "strides": strides, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 3, 5, 5, 3).astype(np.float16)
    ksizes = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    name = None
    input_dict = {"input": input_tensor, "ksizes": ksizes, "strides": strides, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.randint(0, 10, size=(1, 2, 7, 7, 4)).astype(np.uint8)
    ksizes = [1, 3, 3, 3, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "SAME"
    name = "extract_patches_6"
    input_dict = {"input": input_tensor, "ksizes": ksizes, "strides": strides, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.randint(0, 10, size=(2, 1, 9, 9, 5)).astype(np.int64)
    ksizes = [1, 5, 5, 5, 1]
    strides = [1, 4, 4, 4, 1]
    padding = "VALID"
    name = "extract_patches_7"
    input_dict = {"input": input_tensor, "ksizes": ksizes, "strides": strides, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(1, 4, 4, 4, 4).astype(np.float32)
    ksizes = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    name = "extract_patches_8"
    input_dict = {"input": input_tensor, "ksizes": ksizes, "strides": strides, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.randint(0, 10, size=(1, 1, 10, 10, 1)).astype(np.int16)
    ksizes = [1, 3, 3, 3, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "VALID"
    name = "extract_patches_9"
    input_dict = {"input": input_tensor, "ksizes": ksizes, "strides": strides, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(2, 2, 5, 5, 2).astype(np.float16)
    ksizes = [1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    name = "extract_patches_10"
    input_dict = {"input": input_tensor, "ksizes": ksizes, "strides": strides, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.extract_volume_patches"] = tf_extract_volume_patches_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.extract_volume_patches' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.extract_volume_patches'.")

check_valid('tf.extract_volume_patches', generated_inputs['tf.extract_volume_patches'], lib="tf", suffix=0)
