
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_atrous_conv2d_transpose_inputs():
    list_of_inputs = []

    # Input 1
    value = np.random.rand(1, 10, 10, 3).astype(np.float32)
    filters = np.random.rand(3, 3, 3, 3).astype(np.float32)
    output_shape = np.array([1, 14, 14, 3]).astype(np.int32)
    rate = 2
    padding = 'SAME'
    name = 'atrous_conv2d_transpose_1'

    input_dict = {
        "value": value,
        "filters": filters,
        "output_shape": output_shape,
        "rate": rate,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.random.rand(2, 8, 8, 4).astype(np.float32)
    filters = np.random.rand(2, 2, 4, 4).astype(np.float32)
    output_shape = np.array([2, 11, 11, 4]).astype(np.int32)
    rate = 3
    padding = 'VALID'

    input_dict = {
        "value": value,
        "filters": filters,
        "output_shape": output_shape,
        "rate": rate,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.random.rand(4, 16, 16, 2).astype(np.float32)
    filters = np.random.rand(5, 5, 2, 2).astype(np.float32)
    output_shape = np.array([4, 24, 24, 2]).astype(np.int32)
    rate = 4
    padding = 'SAME'

    input_dict = {
        "value": value,
        "filters": filters,
        "output_shape": output_shape,
        "rate": rate,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.random.rand(1, 5, 5, 1).astype(np.float32)
    filters = np.random.rand(4, 4, 1, 1).astype(np.float32)
    output_shape = np.array([1, 8, 8, 1]).astype(np.int32)
    rate = 1
    padding = 'VALID'

    input_dict = {
        "value": value,
        "filters": filters,
        "output_shape": output_shape,
        "rate": rate,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.random.rand(3, 12, 12, 7).astype(np.float32)
    filters = np.random.rand(1, 1, 7, 7).astype(np.float32)
    output_shape = np.array([3, 16, 16, 7]).astype(np.int32)
    rate = 5
    padding = 'SAME'

    input_dict = {
        "value": value,
        "filters": filters,
        "output_shape": output_shape,
        "rate": rate,
        "padding": padding
    }
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
