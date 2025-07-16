
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv2DBackpropInput_inputs():
    list_of_inputs = []

    # Input 1
    input_sizes = np.array([1, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 16).astype(np.float32)
    out_backprop = np.random.rand(1, 8, 8, 16).astype(np.float32)
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
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_sizes = np.array([4, 20, 20, 64], dtype=np.int32)
    filter_val = np.random.rand(5, 5, 64, 32).astype(np.float32)
    out_backprop = np.random.rand(4, 16, 16, 32).astype(np.float32)
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
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_sizes = np.array([1, 16, 16, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 16).astype(np.float32)
    out_backprop = np.random.rand(1, 7, 7, 16).astype(np.float32)
    strides = [1, 2, 2, 1]
    padding = "VALID"
    use_cudnn_on_gpu = True
    explicit_paddings = []
    data_format = "NCHW"
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
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_sizes = np.array([1, 32, 32, 1], dtype=np.int32)
    filter_val = np.random.rand(5, 5, 1, 32).astype(np.float32)
    out_backprop = np.random.rand(1, 16, 16, 32).astype(np.float32)
    strides = [1, 2, 2, 1]
    padding = "SAME"
    use_cudnn_on_gpu = True
    explicit_paddings = []
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
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_sizes = np.array([2, 28, 28, 64], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 64, 32).astype(np.float32)
    out_backprop = np.random.rand(2, 13, 13, 32).astype(np.float32)
    strides = [1, 2, 2, 1]
    padding = "VALID"
    use_cudnn_on_gpu = False
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_backprop_input_6"

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
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_sizes = np.array([1, 24, 24, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 16).astype(np.float32)
    out_backprop = np.random.rand(1, 11, 11, 16).astype(np.float32)
    strides = [1, 2, 2, 1]
    padding = "VALID"
    use_cudnn_on_gpu = True
    explicit_paddings = []
    data_format = "NCHW"
    dilations = [1, 1, 1, 1]
    name = "conv2d_backprop_input_7"

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
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 9
    input_sizes = np.array([1, 64, 64, 128], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 128, 64).astype(np.float32)
    out_backprop = np.random.rand(1, 62, 62, 64).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    use_cudnn_on_gpu = True
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 1, 1, 1]
    name = "conv2d_backprop_input_9"

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
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_sizes = np.array([8, 128, 128, 3], dtype=np.int32)
    filter_val = np.random.rand(7, 7, 3, 32).astype(np.float32)
    out_backprop = np.random.rand(8, 61, 61, 32).astype(np.float32)
    strides = [1, 2, 2, 1]
    padding = "VALID"
    use_cudnn_on_gpu = False
    explicit_paddings = []
    data_format = "NCHW"
    dilations = [1, 1, 1, 1]
    name = "conv2d_backprop_input_10"

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
        "name": name,
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
