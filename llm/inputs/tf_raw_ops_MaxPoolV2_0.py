
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPoolV2_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    ksize_tensor = np.array([1, 2, 2, 1], dtype=np.int32)
    strides_tensor = np.array([1, 1, 1, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize_tensor,
        "strides": strides_tensor,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(1, 10, 10, 1).astype(np.float32)
    ksize_tensor = np.array([1, 3, 3, 1], dtype=np.int32)
    strides_tensor = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "SAME"
    data_format = "NHWC"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize_tensor,
        "strides": strides_tensor,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(2, 8, 8, 32).astype(np.float32)
    ksize_tensor = np.array([1, 4, 4, 1], dtype=np.int32)
    strides_tensor = np.array([1, 4, 4, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize_tensor,
        "strides": strides_tensor,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(2, 32, 32, 3).astype(np.float32)
    ksize_tensor = np.array([1, 2, 2, 1], dtype=np.int32)
    strides_tensor = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "SAME"
    data_format = "NHWC"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize_tensor,
        "strides": strides_tensor,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 3, 3, 1).astype(np.float32)
    ksize_tensor = np.array([1, 3, 3, 1], dtype=np.int32)
    strides_tensor = np.array([1, 1, 1, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize_tensor,
        "strides": strides_tensor,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(1, 3, 3, 1).astype(np.float64)
    ksize_tensor = np.array([1, 3, 3, 1], dtype=np.int32)
    strides_tensor = np.array([1, 1, 1, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize_tensor,
        "strides": strides_tensor,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(1, 3, 3, 1).astype(np.int32)
    ksize_tensor = np.array([1, 3, 3, 1], dtype=np.int32)
    strides_tensor = np.array([1, 1, 1, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize_tensor,
        "strides": strides_tensor,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    input_tensor = np.random.rand(2, 4, 4, 2).astype(np.float32)
    ksize_tensor = np.array([1, 2, 2, 1], dtype=np.int32)
    strides_tensor = np.array([1, 1, 1, 1], dtype=np.int32)
    padding = "SAME"
    data_format = "NCHW"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize_tensor,
        "strides": strides_tensor,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(1, 3, 3, 1).astype(np.int8)
    ksize_tensor = np.array([1, 3, 3, 1], dtype=np.int32)
    strides_tensor = np.array([1, 1, 1, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize_tensor,
        "strides": strides_tensor,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10.
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float16)
    ksize_tensor = np.array([1, 2, 2, 1], dtype=np.int32)
    strides_tensor = np.array([1, 1, 1, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize_tensor,
        "strides": strides_tensor,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
