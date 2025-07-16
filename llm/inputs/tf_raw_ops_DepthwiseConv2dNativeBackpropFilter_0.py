
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DepthwiseConv2dNativeBackpropFilter_inputs():
    list_of_inputs = []

    # Input 1
    input_val = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filter_sizes_val = np.array([3, 3, 3, 1], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 8, 8, 1).astype(np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    explicit_paddings_val = []
    data_format_val = "NHWC"
    dilations_val = [1, 1, 1, 1]
    name_val = None

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "explicit_paddings": explicit_paddings_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DepthwiseConv2dNativeBackpropFilter"] = []

for input_dict in tf_raw_ops_DepthwiseConv2dNativeBackpropFilter_inputs():
    kwargs = input_dict
    generated_inputs["tf.raw_ops.DepthwiseConv2dNativeBackpropFilter"].append(kwargs)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DepthwiseConv2dNativeBackpropFilter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DepthwiseConv2dNativeBackpropFilter'.")

check_valid('tf.raw_ops.DepthwiseConv2dNativeBackpropFilter', generated_inputs['tf.raw_ops.DepthwiseConv2dNativeBackpropFilter'], lib="tf", suffix=0)
