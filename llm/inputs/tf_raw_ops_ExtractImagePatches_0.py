
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_extract_image_patches_inputs():
    list_of_inputs = []

    # Input 1
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
    ksizes = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = None
    input_dict = {"images": images, "ksizes": ksizes, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    images = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    ksizes = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "test_op"
    input_dict = {"images": images, "ksizes": ksizes, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    images = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.float32)
    ksizes = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = None
    input_dict = {"images": images, "ksizes": ksizes, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    images = np.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=np.float32)
    ksizes = [1, 3, 3, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = None
    input_dict = {"images": images, "ksizes": ksizes, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    images = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.float32)
    ksizes = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = None
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
