
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPoolGradGradV2_inputs():
    list_of_inputs = []

    # Input 1
    orig_input = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]], dtype=np.float32)
    orig_output = np.array([[[[5.0, 6.0]]]], dtype=np.float32)
    grad = np.array([[[[0.1, 0.2]]]], dtype=np.float32)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 1, 1, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"
    name = "max_pool_grad_grad_1"

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

    # Input 2
    orig_input = np.array([[[[1, 2], [3, 4]]]], dtype=np.int32)
    orig_output = np.array([[[[4]]]], dtype=np.int32)
    grad = np.array([[[[0.1]]]], dtype=np.int32)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"
    name = "max_pool_grad_grad_2"

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

    # Input 3
    orig_input = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    orig_output = np.array([[[[3.0, 4.0]]]], dtype=np.float32)
    grad = np.array([[[[0.1, 0.2]]]], dtype=np.float32)
    ksize = np.array([1, 1, 1, 1], dtype=np.int32)
    strides = np.array([1, 1, 1, 1], dtype=np.int32)
    padding = "SAME"
    data_format = "NHWC"
    name = "max_pool_grad_grad_3"

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

    # Input 4
    orig_input = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    orig_output = np.array([[[[4.0]]]], dtype=np.float32)
    grad = np.array([[[[0.1]]]], dtype=np.float32)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "SAME"
    data_format = "NHWC"
    name = "max_pool_grad_grad_4"

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

     # Input 5
    orig_input = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float64)
    orig_output = np.array([[[[4.0]]]], dtype=np.float64)
    grad = np.array([[[[0.1]]]], dtype=np.float64)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"
    name = "max_pool_grad_grad_5"

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

    # Input 6
    orig_input = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float16)
    orig_output = np.array([[[[4.0]]]], dtype=np.float16)
    grad = np.array([[[[0.1]]]], dtype=np.float16)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"
    name = "max_pool_grad_grad_6"

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

    # Input 7
    orig_input = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]], dtype=np.float32)
    orig_output = np.array([[[[5.0, 6.0]]]], dtype=np.float32)
    grad = np.array([[[[0.1, 0.2]]]], dtype=np.float32)
    ksize = np.array([1, 2, 2, 1], dtype=np.int32)
    strides = np.array([1, 1, 1, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NCHW"
    name = "max_pool_grad_grad_7"

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

    # Input 8: Different batch size and channel count.
    orig_input = np.random.rand(2, 5, 5, 3).astype(np.float32)
    orig_output = np.random.rand(2, 2, 2, 3).astype(np.float32)
    grad = np.random.rand(2, 2, 2, 3).astype(np.float32)
    ksize = np.array([1, 3, 3, 1], dtype=np.int32)
    strides = np.array([1, 2, 2, 1], dtype=np.int32)
    padding = "VALID"
    data_format = "NHWC"
    name = "max_pool_grad_grad_8"

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

    # Input 9: Different ksize and strides with SAME padding.
    orig_input = np.random.rand(1, 7, 7, 1).astype(np.float32)
    orig_output = np.random.rand(1, 7, 7, 1).astype(np.float32)
    grad = np.random.rand(1, 7, 7, 1).astype(np.float32)
    ksize = np.array([1, 3, 3, 1], dtype=np.int32)
    strides = np.array([1, 1, 1, 1], dtype=np.int32)
    padding = "SAME"
    data_format = "NHWC"
    name = "max_pool_grad_grad_9"

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

    # Input 10: NCHW format.
    orig_input = np.random.rand(1, 3, 7, 7).astype(np.float32)
    orig_output = np.random.rand(1, 3, 4, 4).astype(np.float32)
    grad = np.random.rand(1, 3, 4, 4).astype(np.float32)
    ksize = np.array([1, 1, 3, 3], dtype=np.int32)
    strides = np.array([1, 1, 2, 2], dtype=np.int32)
    padding = "VALID"
    data_format = "NCHW"
    name = "max_pool_grad_grad_10"

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
generated_inputs["tf.raw_ops.MaxPoolGradGradV2"] = tf_raw_ops_MaxPoolGradGradV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MaxPoolGradGradV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolGradGradV2'.")

check_valid('tf.raw_ops.MaxPoolGradGradV2', generated_inputs['tf.raw_ops.MaxPoolGradGradV2'], lib="tf", suffix=0)
