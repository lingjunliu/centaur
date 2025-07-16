
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_dilation2dbackpropfilter_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3).astype(np.float32)
    out_backprop_tensor = np.random.rand(1, 3, 3, 3).astype(np.float32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor,
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
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.randint(0, 255, size=(1, 20, 20, 3), dtype=np.int32)
    filter_tensor = np.random.randint(0, 255, size=(7, 7, 3), dtype=np.int32)
    out_backprop_tensor = np.random.randint(0, 255, size=(1, 14, 14, 3), dtype=np.int32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.randint(0, 100, size=(2, 30, 30, 1), dtype=np.uint8)
    filter_tensor = np.random.randint(0, 100, size=(11, 11, 1), dtype=np.uint8)
    out_backprop_tensor = np.random.randint(0, 100, size=(2, 20, 20, 1), dtype=np.uint8)
    strides = [1, 3, 3, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 8, 8, 2).astype(np.float32)
    filter_tensor = np.random.rand(4, 4, 2).astype(np.float32)
    out_backprop_tensor = np.random.rand(1, 5, 5, 2).astype(np.float32)
    strides = [1, 1, 1, 1]
    rates = [1, 2, 2, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    input_tensor = np.random.rand(3, 12, 12, 4).astype(np.float32)
    filter_tensor = np.random.rand(6, 6, 4).astype(np.float32)
    out_backprop_tensor = np.random.rand(3, 12, 12, 4).astype(np.float32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(1, 15, 15, 1).astype(np.float64)
    filter_tensor = np.random.rand(8, 8, 1).astype(np.float64)
    out_backprop_tensor = np.random.rand(1, 8, 8, 1).astype(np.float64)
    strides = [1, 1, 1, 1]
    rates = [1, 2, 2, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(1, 16, 16, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3).astype(np.float32)
    out_backprop_tensor = np.random.rand(1, 8, 8, 3).astype(np.float32)
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(2, 20, 20, 2).astype(np.float32)
    filter_tensor = np.random.rand(5, 5, 2).astype(np.float32)
    out_backprop_tensor = np.random.rand(2, 10, 10, 2).astype(np.float32)
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(1, 25, 25, 1).astype(np.float64)
    filter_tensor = np.random.rand(5, 5, 1).astype(np.float64)
    out_backprop_tensor = np.random.rand(1, 25, 25, 1).astype(np.float64)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Dilation2DBackpropFilter"] = tf_raw_ops_dilation2dbackpropfilter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Dilation2DBackpropFilter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Dilation2DBackpropFilter'.")

check_valid('tf.raw_ops.Dilation2DBackpropFilter', generated_inputs['tf.raw_ops.Dilation2DBackpropFilter'], lib="tf", suffix=0)
