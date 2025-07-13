
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_maxpoolgradv2_inputs():
    list_of_inputs = []

    # Input 1, valid NHWC
    orig_input = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    orig_output = np.array([[[[4.0]]]], dtype=np.float32)
    grad = np.array([[[[1.0]]]], dtype=np.float32)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"
    name = None

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid NCHW
    orig_input = np.array([[[[[1.0], [2.0]], [[3.0], [4.0]]]]], dtype=np.float32)
    orig_output = np.array([[[[[4.0]]]]], dtype=np.float32)
    grad = np.array([[[[[1.0]]]]], dtype=np.float32)
    ksize = np.array([1, 1, 2, 2], dtype=np.int32)
    strides = np.array([1, 1, 2, 2], dtype=np.int32)
    padding = "VALID"
    data_format = "NCHW"
    name = None

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, SAME padding
    orig_input = np.array([[[[1.0], [2.0], [5.0]], [[3.0], [4.0], [6.0]], [[7.0], [8.0], [9.0]]]], dtype=np.float32)
    orig_output = np.array([[[[4.0, 6.0]], [[8.0, 9.0]]]], dtype=np.float32)
    grad = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=np.float32)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "SAME"
    data_format = "NHWC"
    name = None

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, different ksize and strides
    orig_input = np.array([[[[1.0], [2.0], [5.0]], [[3.0], [4.0], [6.0]], [[7.0], [8.0], [9.0]]]], dtype=np.float32)
    orig_output = np.array([[[[9.0]]]], dtype=np.float32)
    grad = np.array([[[[1.0]]]], dtype=np.float32)
    ksize = np.array([1, 3, 3, 1], dtype=np.int32)
    strides = np.array([1, 1, 1, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"
    name = None

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, multiple batches
    orig_input = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]], [[[5.0], [6.0]], [[7.0], [8.0]]]], dtype=np.float32)
    orig_output = np.array([[[[4.0]], [[6.0]]]], dtype=np.float32)
    grad = np.array([[[[1.0]], [[2.0]]]], dtype=np.float32)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"
    name = None

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6, int32 input
    orig_input = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int32)
    orig_output = np.array([[[[4]]]], dtype=np.int32)
    grad = np.array([[[[1]]]], dtype=np.int32)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"
    name = None

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, negative gradient values
    orig_input = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    orig_output = np.array([[[[4.0]]]], dtype=np.float32)
    grad = np.array([[[[-1.0]]]], dtype=np.float32)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"
    name = None

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, different float type
    orig_input = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float64)
    orig_output = np.array([[[[4.0]]]], dtype=np.float64)
    grad = np.array([[[[1.0]]]], dtype=np.float64)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"
    name = None

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, uint8 type
    orig_input = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint8)
    orig_output = np.array([[[[4]]]], dtype=np.uint8)
    grad = np.array([[[[1]]]], dtype=np.uint8)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"
    name = None

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
        "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, half type
    orig_input = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float16)
    orig_output = np.array([[[[4.0]]]], dtype=np.float16)
    grad = np.array([[[[1.0]]]], dtype=np.float16)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"
    name = None

    input_dict = {
        "orig_input": orig_input,
        "orig_output": orig_output,
    "grad": grad,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_maxpoolgradv2_inputs()
generated_inputs["tf.raw_ops.MaxPoolGradV2"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.MaxPoolGradV2"].append(input_dict)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MaxPoolGradV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolGradV2'.")

check_valid('tf.raw_ops.MaxPoolGradV2', generated_inputs['tf.raw_ops.MaxPoolGradV2'], lib="tf", suffix=0)
