
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_atrous_conv2d_transpose_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    value = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filters = np.random.rand(3, 3, 3, 3).astype(np.float32)
    output_shape = np.array([1, 12, 12, 3], dtype=np.int32)
    rate = 1
    padding = "VALID"
    name = "deconv1"
    input_dict = {"value": value, "filters": filters, "output_shape": output_shape, "rate": rate, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: SAME padding
    value = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filters = np.random.rand(3, 3, 3, 3).astype(np.float32)
    output_shape = np.array([1, 10, 10, 3], dtype=np.int32)
    rate = 1
    padding = "SAME"
    name = "deconv2"
    input_dict = {"value": value, "filters": filters, "output_shape": output_shape, "rate": rate, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different batch size
    value = np.random.rand(4, 10, 10, 3).astype(np.float32)
    filters = np.random.rand(3, 3, 3, 3).astype(np.float32)
    output_shape = np.array([4, 12, 12, 3], dtype=np.int32)
    rate = 1
    padding = "VALID"
    name = "deconv3"
    input_dict = {"value": value, "filters": filters, "output_shape": output_shape, "rate": rate, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.atrous_conv2d_transpose"] = tf_nn_atrous_conv2d_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.atrous_conv2d_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.atrous_conv2d_transpose'.")

check_valid('tf.nn.atrous_conv2d_transpose', generated_inputs['tf.nn.atrous_conv2d_transpose'], lib="tf", suffix=0)
