
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_extract_image_patches_inputs():
    list_of_inputs = []

    # Input 1
    images = np.random.rand(1, 10, 10, 3).astype(np.float32)
    ksizes = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "extract_patches_1"
    input_dict = {"images": images, "ksizes": ksizes, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    images = np.random.rand(2, 20, 20, 1).astype(np.float32)
    ksizes = [1, 5, 5, 1]
    strides = [1, 3, 3, 1]
    rates = [1, 2, 2, 1]
    padding = "SAME"
    name = "extract_patches_2"
    input_dict = {"images": images, "ksizes": ksizes, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    images = np.random.rand(1, 32, 32, 3).astype(np.float32)
    ksizes = [1, 7, 7, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "extract_patches_3"
    input_dict = {"images": images, "ksizes": ksizes, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    images = np.random.rand(4, 16, 16, 4).astype(np.float32)
    ksizes = [1, 4, 4, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 3, 3, 1]
    padding = "SAME"
    name = "extract_patches_4"
    input_dict = {"images": images, "ksizes": ksizes, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    images = np.random.rand(1, 28, 28, 1).astype(np.float32)
    ksizes = [1, 3, 3, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "extract_patches_5"
    input_dict = {"images": images, "ksizes": ksizes, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    images = np.random.rand(2, 64, 64, 3).astype(np.float32)
    ksizes = [1, 8, 8, 1]
    strides = [1, 4, 4, 1]
    rates = [1, 2, 2, 1]
    padding = "SAME"
    name = "extract_patches_6"
    input_dict = {"images": images, "ksizes": ksizes, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    images = np.random.rand(1, 10, 10, 3).astype(np.float64)
    ksizes = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "extract_patches_7"
    input_dict = {"images": images, "ksizes": ksizes, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    images = np.random.rand(2, 20, 20, 1).astype(np.int32)
    ksizes = [1, 5, 5, 1]
    strides = [1, 3, 3, 1]
    rates = [1, 2, 2, 1]
    padding = "SAME"
    name = "extract_patches_8"
    input_dict = {"images": images, "ksizes": ksizes, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    images = np.random.rand(1, 32, 32, 3).astype(np.uint8)
    ksizes = [1, 7, 7, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "extract_patches_9"
    input_dict = {"images": images, "ksizes": ksizes, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    images = np.random.rand(4, 16, 16, 4).astype(np.complex64)
    ksizes = [1, 4, 4, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 3, 3, 1]
    padding = "SAME"
    name = "extract_patches_10"
    input_dict = {"images": images, "ksizes": ksizes, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    images = np.random.rand(1, 28, 28, 1).astype(np.bool_)
    ksizes = [1, 3, 3, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "extract_patches_11"
    input_dict = {"images": images, "ksizes": ksizes, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ExtractImagePatches"] = tf_raw_ops_extract_image_patches_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ExtractImagePatches' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ExtractImagePatches'.")

check_valid('tf.raw_ops.ExtractImagePatches', generated_inputs['tf.raw_ops.ExtractImagePatches'], lib="tf", suffix=0)
