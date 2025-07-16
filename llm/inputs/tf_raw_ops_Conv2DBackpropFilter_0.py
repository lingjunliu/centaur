
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv2DBackpropFilter_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 28, 28, 3).astype(np.float32)
    filter_sizes1 = np.array([5, 5, 3, 64], dtype=np.int32)
    out_backprop1 = np.random.rand(1, 24, 24, 64).astype(np.float32)
    strides1 = [1, 1, 1, 1]
    padding1 = "VALID"
    use_cudnn_on_gpu1 = True
    explicit_paddings1 = []
    data_format1 = "NHWC"
    dilations1 = [1, 1, 1, 1]
    name1 = "conv2d_backprop_filter_1"

    input_dict1 = {
        "input": input1,
        "filter_sizes": filter_sizes1,
        "out_backprop": out_backprop1,
        "strides": strides1,
        "padding": padding1,
        "use_cudnn_on_gpu": use_cudnn_on_gpu1,
        "explicit_paddings": explicit_paddings1,
        "data_format": data_format1,
        "dilations": dilations1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(2, 32, 32, 1).astype(np.float32)
    filter_sizes2 = np.array([3, 3, 1, 32], dtype=np.int32)
    out_backprop2 = np.random.rand(2, 30, 30, 32).astype(np.float32)
    strides2 = [1, 1, 1, 1]
    padding2 = "VALID"
    use_cudnn_on_gpu2 = False
    explicit_paddings2 = []
    data_format2 = "NHWC"
    dilations2 = [1, 1, 1, 1]
    name2 = "conv2d_backprop_filter_2"

    input_dict2 = {
        "input": input2,
        "filter_sizes": filter_sizes2,
        "out_backprop": out_backprop2,
        "strides": strides2,
        "padding": padding2,
        "use_cudnn_on_gpu": use_cudnn_on_gpu2,
        "explicit_paddings": explicit_paddings2,
        "data_format": data_format2,
        "dilations": dilations2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.random.rand(1, 16, 16, 64).astype(np.float32)
    filter_sizes3 = np.array([7, 7, 64, 128], dtype=np.int32)
    out_backprop3 = np.random.rand(1, 8, 8, 128).astype(np.float32)
    strides3 = [1, 2, 2, 1]
    padding3 = "VALID"
    use_cudnn_on_gpu3 = True
    explicit_paddings3 = []
    data_format3 = "NHWC"
    dilations3 = [1, 1, 1, 1]
    name3 = "conv2d_backprop_filter_3"

    input_dict3 = {
        "input": input3,
        "filter_sizes": filter_sizes3,
        "out_backprop": out_backprop3,
        "strides": strides3,
        "padding": padding3,
        "use_cudnn_on_gpu": use_cudnn_on_gpu3,
        "explicit_paddings": explicit_paddings3,
        "data_format": data_format3,
        "dilations": dilations3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Conv2DBackpropFilter"] = tf_raw_ops_Conv2DBackpropFilter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Conv2DBackpropFilter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv2DBackpropFilter'.")

check_valid('tf.raw_ops.Conv2DBackpropFilter', generated_inputs['tf.raw_ops.Conv2DBackpropFilter'], lib="tf", suffix=0)
