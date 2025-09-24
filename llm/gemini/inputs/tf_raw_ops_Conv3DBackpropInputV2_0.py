
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv3DBackpropInputV2_inputs():
    list_of_inputs = []

    # Input 1
    input_sizes = np.array([1, 3, 10, 10, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 3, 5).astype(np.float32)
    out_backprop = np.random.rand(1, 1, 8, 8, 5).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv3d_backprop_input_1"

    input_dict = {
        "input_sizes": input_sizes,
        "filter": filter_val,
        "out_backprop": out_backprop,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "dilations": dilations,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Conv3DBackpropInputV2"] = tf_raw_ops_Conv3DBackpropInputV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Conv3DBackpropInputV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv3DBackpropInputV2'.")

check_valid('tf.raw_ops.Conv3DBackpropInputV2', generated_inputs['tf.raw_ops.Conv3DBackpropInputV2'], lib="tf", suffix=0)
