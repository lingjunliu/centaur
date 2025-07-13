
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_conv3d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3, 3, 2).astype(np.float32)
    strides_val = [1, 1, 1, 1, 1]
    padding_val = "VALID"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 1, 1, 1]

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Conv3D"] = tf_raw_ops_conv3d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Conv3D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv3D'.")

check_valid('tf.raw_ops.Conv3D', generated_inputs['tf.raw_ops.Conv3D'], lib="tf", suffix=0)
