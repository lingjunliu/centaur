
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_max_pool_grad_grad_inputs():
    list_of_inputs = []

    # Input 1
    orig_input = np.random.rand(1, 5, 5, 3).astype(np.float32)
    orig_output = np.random.rand(1, 2, 2, 3).astype(np.float32)
    grad = np.random.rand(1, 2, 2, 3).astype(np.float32)
    ksize = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
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

    # Input 2
    orig_input = np.random.rand(2, 10, 10, 1).astype(np.float64)
    orig_output = np.random.rand(2, 5, 5, 1).astype(np.float64)
    grad = np.random.rand(2, 5, 5, 1).astype(np.float64)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
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

    # Input 3
    orig_input = np.random.rand(1, 3, 32, 32).astype(np.float32)
    orig_output = np.random.rand(1, 1, 16, 16).astype(np.float32)
    grad = np.random.rand(1, 1, 16, 16).astype(np.float32)
    ksize = [1, 1, 2, 2]
    strides = [1, 1, 2, 2]
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

    # Input 4
    orig_input = np.random.rand(4, 8, 8, 8).astype(np.float32)
    orig_output = np.random.rand(4, 4, 4, 8).astype(np.float32)
    grad = np.random.rand(4, 4, 4, 8).astype(np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
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

    # Input 5
    orig_input = np.random.rand(1, 16, 16, 16).astype(np.float32)
    orig_output = np.random.rand(1, 8, 8, 16).astype(np.float32)
    grad = np.random.rand(1, 8, 8, 16).astype(np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
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

    # Input 6
    orig_input = np.random.rand(2, 4, 4, 2).astype(np.float32)
    orig_output = np.random.rand(2, 2, 2, 2).astype(np.float32)
    grad = np.random.rand(2, 2, 2, 2).astype(np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
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

   # Input 7
    orig_input = np.random.rand(1, 5, 5, 3).astype(np.float32)
    orig_output = np.random.rand(1, 2, 2, 3).astype(np.float32)
    grad = np.random.rand(1, 2, 2, 3).astype(np.float32)
    ksize = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
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

   # Input 8
    orig_input = np.random.rand(1, 5, 5, 3).astype(np.float32)
    orig_output = np.random.rand(1, 2, 2, 3).astype(np.float32)
    grad = np.random.rand(1, 2, 2, 3).astype(np.float32)
    ksize = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
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

   # Input 9
    orig_input = np.random.rand(1, 5, 5, 3).astype(np.float32)
    orig_output = np.random.rand(1, 2, 2, 3).astype(np.float32)
    grad = np.random.rand(1, 2, 2, 3).astype(np.float32)
    ksize = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
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

    # Input 10
    orig_input = np.random.rand(1, 5, 5, 3).astype(np.float32)
    orig_output = np.random.rand(1, 2, 2, 3).astype(np.float32)
    grad = np.random.rand(1, 2, 2, 3).astype(np.float32)
    ksize = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
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

    # Input 11
    orig_input = np.random.rand(2, 7, 7, 5).astype(np.float32)
    orig_output = np.random.rand(2, 4, 4, 5).astype(np.float32)
    grad = np.random.rand(2, 4, 4, 5).astype(np.float32)
    ksize = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
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

    # Input 12
    orig_input = np.random.rand(1, 7, 7, 1).astype(np.float32)
    orig_output = np.random.rand(1, 7, 7, 1).astype(np.float32)
    grad = np.random.rand(1, 7, 7, 1).astype(np.float32)
    ksize = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
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

    # Input 13
    orig_input = np.random.rand(1, 5, 5, 3).astype(np.float32)
    orig_output = np.random.rand(1, 5, 5, 3).astype(np.float32)
    grad = np.random.rand(1, 5, 5, 3).astype(np.float32)
    ksize = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
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

generated_inputs = {}
generated_inputs["tf.raw_ops.MaxPoolGradGrad"] = tf_raw_ops_max_pool_grad_grad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MaxPoolGradGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolGradGrad'.")

check_valid('tf.raw_ops.MaxPoolGradGrad', generated_inputs['tf.raw_ops.MaxPoolGradGrad'], lib="tf", suffix=0)
