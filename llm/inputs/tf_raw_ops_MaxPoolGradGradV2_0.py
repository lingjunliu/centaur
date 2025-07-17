
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPoolGradGradV2_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    orig_input = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    orig_output = np.array([[[[3.0, 4.0]]]], dtype=np.float32)
    grad = np.array([[[[0.5, 0.6]]]], dtype=np.float32)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: SAME padding
    orig_input = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    orig_output = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    grad = np.array([[[[0.5, 0.6], [0.7, 0.8]]]], dtype=np.float32)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 1, 1, 1], dtype=np.int32)
    padding = "SAME"
    data_format = "NHWC"

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NCHW data format
    orig_input = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    orig_output = np.array([[[[3.0], [4.0]]]], dtype=np.float32)
    grad = np.array([[[[0.5], [0.6]]]], dtype=np.float32)
    ksize = np.array([1, 1, 2, 2], dtype=np.int32)
    strides = np.array([1, 1, 2, 2], dtype=np.int32)
    padding = "VALID"
    data_format = "NCHW"

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple batches
    orig_input = np.array([[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    orig_output = np.array([[[[3.0, 4.0]]], [[[7.0, 8.0]]]], dtype=np.float32)
    grad = np.array([[[[0.5, 0.6]]], [[[0.7, 0.8]]]], dtype=np.float32)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different ksize and strides
    orig_input = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32)
    orig_output = np.array([[[[5.0, 6.0]]]], dtype=np.float32)
    grad = np.array([[[[0.5, 0.6]]]], dtype=np.float32)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 1, 1, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different data type (int32)
    orig_input = np.array([[[[1, 2], [3, 4]]]], dtype=np.int32)
    orig_output = np.array([[[[3, 4]]]], dtype=np.int32)
    grad = np.array([[[[5, 6]]]], dtype=np.int32)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different data type (float16)
    orig_input = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float16)
    orig_output = np.array([[[[3.0, 4.0]]]], dtype=np.float16)
    grad = np.array([[[[0.5, 0.6]]]], dtype=np.float16)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multiple channels
    orig_input = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]], dtype=np.float32)
    orig_output = np.array([[[[3.0, 5.0, 6.0]]]], dtype=np.float32)
    grad = np.array([[[[0.5, 0.6, 0.7]]]], dtype=np.float32)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: ksize and strides = 1
    orig_input = np.array([[[[1.0]]]], dtype=np.float32)
    orig_output = np.array([[[[1.0]]]], dtype=np.float32)
    grad = np.array([[[[0.5]]]], dtype=np.float32)
    ksize = np.array([1, 1, 1, 1], dtype=np.int32)
    strides = np.array([1, 1, 1, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int8 data type
    orig_input = np.array([[[[1, 2], [3, 4]]]], dtype=np.int8)
    orig_output = np.array([[[[3, 4]]]], dtype=np.int8)
    grad = np.array([[[[5, 6]]]], dtype=np.int8)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolGradGradV2"] = tf_raw_ops_MaxPoolGradGradV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MaxPoolGradGradV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolGradGradV2'.")

check_valid('tf.raw_ops.MaxPoolGradGradV2', generated_inputs['tf.raw_ops.MaxPoolGradGradV2'], lib="tf", suffix=0)
