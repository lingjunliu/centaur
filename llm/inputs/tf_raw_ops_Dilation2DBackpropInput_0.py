
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_dilation2d_backprop_input_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3).astype(np.float32)
    out_backprop_tensor = np.random.rand(1, 3, 3, 3).astype(np.float32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor, dtype=tf.float32),
        "filter": tf.convert_to_tensor(filter_tensor, dtype=tf.float32),
        "out_backprop": tf.convert_to_tensor(out_backprop_tensor, dtype=tf.float32),
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 10, 10, 1).astype(np.float64)
    filter_tensor = np.random.rand(5, 5, 1).astype(np.float64)
    out_backprop_tensor = np.random.rand(2, 6, 6, 1).astype(np.float64)
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor, dtype=tf.float64),
        "filter": tf.convert_to_tensor(filter_tensor, dtype=tf.float64),
        "out_backprop": tf.convert_to_tensor(out_backprop_tensor, dtype=tf.float64),
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.randint(0, 10, size=(1, 8, 8, 2), dtype=np.int32)
    filter_tensor = np.random.randint(0, 10, size=(2, 2, 2), dtype=np.int32)
    out_backprop_tensor = np.random.randint(0, 10, size=(1, 8, 8, 2), dtype=np.int32)
    strides = [1, 1, 1, 1]
    rates = [1, 2, 2, 1]
    padding = "SAME"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor, dtype=tf.int32),
        "filter": tf.convert_to_tensor(filter_tensor, dtype=tf.int32),
        "out_backprop": tf.convert_to_tensor(out_backprop_tensor, dtype=tf.int32),
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(4, 12, 12, 4).astype(np.float32)
    filter_tensor = np.random.rand(4, 4, 4).astype(np.float32)
    out_backprop_tensor = np.random.rand(4, 6, 6, 4).astype(np.float32)
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor, dtype=tf.float32),
        "filter": tf.convert_to_tensor(filter_tensor, dtype=tf.float32),
        "out_backprop": tf.convert_to_tensor(out_backprop_tensor, dtype=tf.float32),
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 7, 7, 1).astype(np.float32)
    filter_tensor = np.random.rand(2, 2, 1).astype(np.float32)
    out_backprop_tensor = np.random.rand(1, 7, 7, 1).astype(np.float32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"

    input_dict = {
        "input": tf.convert_to_tensor(input_tensor, dtype=tf.float32),
        "filter": tf.convert_to_tensor(filter_tensor, dtype=tf.float32),
        "out_backprop": tf.convert_to_tensor(out_backprop_tensor, dtype=tf.float32),
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Dilation2DBackpropInput"] = tf_raw_ops_dilation2d_backprop_input_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Dilation2DBackpropInput' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Dilation2DBackpropInput'.")

check_valid('tf.raw_ops.Dilation2DBackpropInput', generated_inputs['tf.raw_ops.Dilation2DBackpropInput'], lib="tf", suffix=0)
