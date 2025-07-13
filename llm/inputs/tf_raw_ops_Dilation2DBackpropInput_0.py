
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_dilation2dbackpropinput_inputs():
    list_of_inputs = []

    # Input 1
    input_np = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter_np = np.random.rand(3, 3, 3).astype(np.float32)
    out_backprop_np = np.random.rand(1, 5, 5, 3).astype(np.float32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation2d_backprop_input_1"

    input_dict = {
        "input": input_np,
        "filter": filter_np,
        "out_backprop": out_backprop_np,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_np = np.random.rand(2, 10, 10, 1).astype(np.float64)
    filter_np = np.random.rand(5, 5, 1).astype(np.float64)
    out_backprop_np = np.random.rand(2, 6, 6, 1).astype(np.float64)
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "dilation2d_backprop_input_2"

    input_dict = {
        "input": input_np,
        "filter": filter_np,
        "out_backprop": out_backprop_np,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_np = np.random.randint(0, 10, size=(1, 7, 7, 2), dtype=np.int32)
    filter_np = np.random.randint(0, 5, size=(3, 3, 2), dtype=np.int32)
    out_backprop_np = np.random.randint(0, 8, size=(1, 7, 7, 2), dtype=np.int32)
    strides = [1, 1, 1, 1]
    rates = [1, 2, 2, 1]
    padding = "SAME"
    name = "dilation2d_backprop_input_3"

    input_dict = {
        "input": input_np,
        "filter": filter_np,
        "out_backprop": out_backprop_np,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_np = np.random.randint(0, 10, size=(2, 8, 8, 3), dtype=np.uint8)
    filter_np = np.random.randint(0, 5, size=(4, 4, 3), dtype=np.uint8)
    out_backprop_np = np.random.randint(0, 8, size=(2, 4, 4, 3), dtype=np.uint8)
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "dilation2d_backprop_input_4"

    input_dict = {
        "input": input_np,
        "filter": filter_np,
        "out_backprop": out_backprop_np,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_np = np.random.randint(0, 10, size=(1, 6, 6, 1), dtype=np.int64)
    filter_np = np.random.randint(0, 5, size=(2, 2, 1), dtype=np.int64)
    out_backprop_np = np.random.randint(0, 8, size=(1, 6, 6, 1), dtype=np.int64)
    strides = [1, 1, 1, 1]
    rates = [1, 3, 3, 1]
    padding = "SAME"
    name = "dilation2d_backprop_input_5"

    input_dict = {
        "input": input_np,
        "filter": filter_np,
        "out_backprop": out_backprop_np,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_np = np.random.rand(1, 5, 5, 3).astype(np.float16)
    filter_np = np.random.rand(3, 3, 3).astype(np.float16)
    out_backprop_np = np.random.rand(1, 1, 1, 3).astype(np.float16)
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "dilation2d_backprop_input_6"

    input_dict = {
        "input": input_np,
        "filter": filter_np,
        "out_backprop": out_backprop_np,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_np = np.random.randint(0, 10, size=(2, 8, 8, 3), dtype=np.uint16)
    filter_np = np.random.randint(0, 5, size=(4, 4, 3), dtype=np.uint16)
    out_backprop_np = np.random.randint(0, 8, size=(2, 8, 8, 3), dtype=np.uint16)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation2d_backprop_input_7"

    input_dict = {
        "input": input_np,
        "filter": filter_np,
        "out_backprop": out_backprop_np,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_np = np.random.rand(1, 5, 5, 3).astype(np.float16)
    filter_np = np.random.rand(3, 3, 3).astype(np.float16)
    out_backprop_np = np.random.rand(1, 5, 5, 3).astype(np.float16)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation2d_backprop_input_8"

    input_dict = {
        "input": input_np,
        "filter": filter_np,
        "out_backprop": out_backprop_np,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_np = np.random.randint(0, 10, size=(2, 8, 8, 3), dtype=np.uint32)
    filter_np = np.random.randint(0, 5, size=(4, 4, 3), dtype=np.uint32)
    out_backprop_np = np.random.randint(0, 8, size=(2, 8, 8, 3), dtype=np.uint32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation2d_backprop_input_9"

    input_dict = {
        "input": input_np,
        "filter": filter_np,
        "out_backprop": out_backprop_np,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    input_np = np.random.randint(0, 10, size=(2, 8, 8, 3), dtype=np.uint64)
    filter_np = np.random.randint(0, 5, size=(4, 4, 3), dtype=np.uint64)
    out_backprop_np = np.random.randint(0, 8, size=(2, 8, 8, 3), dtype=np.uint64)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation2d_backprop_input_10"

    input_dict = {
        "input": input_np,
        "filter": filter_np,
        "out_backprop": out_backprop_np,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Dilation2DBackpropInput"] = tf_raw_ops_dilation2dbackpropinput_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Dilation2DBackpropInput' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Dilation2DBackpropInput'.")

check_valid('tf.raw_ops.Dilation2DBackpropInput', generated_inputs['tf.raw_ops.Dilation2DBackpropInput'], lib="tf", suffix=0)
