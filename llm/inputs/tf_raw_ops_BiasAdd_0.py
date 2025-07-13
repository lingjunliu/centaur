
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_bias_add_inputs():
    list_of_inputs = []

    # Input 1: NHWC, float32
    value = np.random.rand(1, 28, 28, 3).astype(np.float32)
    bias = np.random.rand(3).astype(np.float32)
    data_format = "NHWC"
    name = "bias_add_1"
    input_dict = {"value": value, "bias": bias, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NCHW, float32
    value = np.random.rand(1, 3, 28, 28).astype(np.float32)
    bias = np.random.rand(3).astype(np.float32)
    data_format = "NCHW"
    name = "bias_add_2"
    input_dict = {"value": value, "bias": bias, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NHWC, int32
    value = np.random.randint(0, 10, size=(2, 10, 10, 5), dtype=np.int32)
    bias = np.random.randint(0, 10, size=(5), dtype=np.int32)
    data_format = "NHWC"
    name = "bias_add_3"
    input_dict = {"value": value, "bias": bias, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BiasAdd"] = tf_raw_ops_bias_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BiasAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BiasAdd'.")

check_valid('tf.raw_ops.BiasAdd', generated_inputs['tf.raw_ops.BiasAdd'], lib="tf", suffix=0)
