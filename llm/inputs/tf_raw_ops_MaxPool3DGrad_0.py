
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_max_pool3d_grad_inputs():
    list_of_inputs = []

    # Input 1
    orig_input = np.random.rand(1, 3, 3, 3, 1).astype(np.float32)
    orig_output = np.random.rand(1, 2, 2, 2, 1).astype(np.float32)
    grad = np.random.rand(1, 2, 2, 2, 1).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_grad_1"

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
    orig_input = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    orig_output = np.random.rand(1, 2, 2, 2, 3).astype(np.float32)
    grad = np.random.rand(1, 2, 2, 2, 3).astype(np.float32)
    ksize = [1, 3, 3, 3, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_grad_2"
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
    orig_input = np.random.rand(2, 4, 4, 4, 2).astype(np.float32)
    orig_output = np.random.rand(2, 4, 4, 4, 2).astype(np.float32)
    grad = np.random.rand(2, 4, 4, 4, 2).astype(np.float32)
    ksize = [1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = "max_pool_grad_3"
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
    orig_input = np.random.rand(1, 5, 5, 5, 1).astype(np.float32)
    orig_output = np.random.rand(1, 5, 5, 5, 1).astype(np.float32)
    grad = np.random.rand(1, 5, 5, 5, 1).astype(np.float32)
    ksize = [1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = "max_pool_grad_4"
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
    orig_input = np.random.rand(1, 3, 3, 3, 1).astype(np.float32)
    orig_output = np.random.rand(1, 1, 1, 1, 1).astype(np.float32)
    grad = np.random.rand(1, 1, 1, 1, 1).astype(np.float32)
    ksize = [1, 3, 3, 3, 1]
    strides = [1, 3, 3, 3, 1]
    padding = "VALID"
    data_format = "NDHWC"
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

    # Input 6
    orig_input = np.random.rand(2, 3, 3, 3, 2).astype(np.float32)
    orig_output = np.random.rand(2, 2, 2, 2, 2).astype(np.float32)
    grad = np.random.rand(2, 2, 2, 2, 2).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_grad_6"
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
    orig_input = np.random.rand(1, 4, 4, 4, 1).astype(np.float32)
    orig_output = np.random.rand(1, 2, 2, 2, 1).astype(np.float32)
    grad = np.random.rand(1, 2, 2, 2, 1).astype(np.float32)
    ksize = [1, 3, 3, 3, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_grad_7"

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

    # Input 8
    orig_input = np.random.rand(1, 1, 5, 5, 5).astype(np.float32)
    orig_output = np.random.rand(1, 1, 3, 3, 3).astype(np.float32)
    grad = np.random.rand(1, 1, 3, 3, 3).astype(np.float32)
    ksize = [1, 1, 3, 3, 3]
    strides = [1, 1, 2, 2, 2]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_grad_8"

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

    # Input 9
    orig_input = np.random.rand(2, 2, 4, 4, 4).astype(np.float32)
    orig_output = np.random.rand(2, 2, 4, 4, 4).astype(np.float32)
    grad = np.random.rand(2, 2, 4, 4, 4).astype(np.float32)
    ksize = [1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = "max_pool_grad_9"

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

    # Input 10
    orig_input = np.random.rand(1, 1, 3, 3, 3).astype(np.float32)
    orig_output = np.random.rand(1, 1, 1, 1, 1).astype(np.float32)
    grad = np.random.rand(1, 1, 1, 1, 1).astype(np.float32)
    ksize = [1, 1, 3, 3, 3]
    strides = [1, 1, 3, 3, 3]
    padding = "VALID"
    data_format = "NDHWC"
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
generated_inputs["tf.raw_ops.MaxPool3DGrad"] = tf_raw_ops_max_pool3d_grad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MaxPool3DGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPool3DGrad'.")

check_valid('tf.raw_ops.MaxPool3DGrad', generated_inputs['tf.raw_ops.MaxPool3DGrad'], lib="tf", suffix=0)
