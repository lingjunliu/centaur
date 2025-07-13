
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv2DBackpropInput_inputs():
    list_of_inputs = []

    # Input 1
    input_sizes = np.array([1, 28, 28, 3], dtype=np.int32)
    filter_val = np.random.rand(5, 5, 3, 16).astype(np.float32)
    out_backprop = np.random.rand(1, 24, 24, 16).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    use_cudnn_on_gpu = True
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_backprop_input_1"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_sizes = np.array([4, 32, 32, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 32).astype(np.float32)
    out_backprop = np.random.rand(4, 32, 32, 32).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    use_cudnn_on_gpu = False
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_backprop_input_2"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_sizes = np.array([1, 64, 64, 1], dtype=np.int32)
    filter_val = np.random.rand(7, 7, 1, 64).astype(np.float32)
    out_backprop = np.random.rand(1, 32, 32, 64).astype(np.float32)
    strides = [1, 2, 2, 1]
    padding = "VALID"
    use_cudnn_on_gpu = True
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_backprop_input_3"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4
    input_sizes = np.array([2, 16, 16, 8], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 8, 16).astype(np.float32)
    out_backprop = np.random.rand(2, 8, 8, 16).astype(np.float32)
    strides = [1, 2, 2, 1]
    padding = "SAME"
    use_cudnn_on_gpu = True
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_backprop_input_4"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Explicit padding
    input_sizes = np.array([1, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 16).astype(np.float32)
    out_backprop = np.random.rand(1, 12, 12, 16).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "EXPLICIT"
    use_cudnn_on_gpu = True
    explicit_paddings = [0, 2, 0, 2]
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_backprop_input_5"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "use_cudnn_on_gpu": use_cudnn_on_gpu,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Conv2DBackpropInput"] = tf_raw_ops_Conv2DBackpropInput_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Conv2DBackpropInput' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv2DBackpropInput'.")

check_valid('tf.raw_ops.Conv2DBackpropInput', generated_inputs['tf.raw_ops.Conv2DBackpropInput'], lib="tf", suffix=0)
