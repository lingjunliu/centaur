
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_Dilation2D_inputs():
    list_of_inputs = []
    
    # Input 1
    input_val = np.random.randn(1, 3, 3, 1).astype(np.float32)
    filter_val = np.random.randn(2, 2, 1).astype(np.float32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "dilation_1"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 2
    input_val = np.random.randn(2, 4, 4, 3).astype(np.float64)
    filter_val = np.random.randn(3, 3, 3).astype(np.float64)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation_2"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 3
    input_val = np.random.randint(-10, 10, size=(1, 5, 5, 2)).astype(np.int32)
    filter_val = np.random.randint(-10, 10, size=(2, 2, 2)).astype(np.int32)
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "dilation_3"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 4
    input_val = np.random.randn(1, 6, 6, 1).astype(np.float32)
    filter_val = np.random.randn(2, 2, 1).astype(np.float32)
    strides = [1, 1, 1, 1]
    rates = [1, 2, 2, 1]
    padding = "SAME"
    name = "dilation_4"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 5
    input_val = np.random.randint(-10, 10, size=(1, 3, 3, 1)).astype(np.int32)
    filter_val = np.random.randint(-10, 10, size=(1, 1, 1)).astype(np.int32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "dilation_5"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 6
    input_val = np.random.randn(2, 5, 5, 4).astype(np.float64)
    filter_val = np.random.randn(3, 3, 4).astype(np.float64)
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation_6"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 7
    input_val = np.random.randint(-50, 50, size=(1, 7, 7, 1)).astype(np.int32)
    filter_val = np.random.randint(-50, 50, size=(2, 2, 1)).astype(np.int32)
    strides = [1, 1, 1, 1]
    rates = [1, 3, 3, 1]
    padding = "VALID"
    name = "dilation_7"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 8
    input_val = np.random.randn(1, 4, 4, 2).astype(np.float32)
    filter_val = np.random.randn(2, 2, 2).astype(np.float32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation_8"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 9
    input_val = np.random.randn(1, 3, 3, 1).astype(np.float64)
    filter_val = np.random.randn(2, 2, 1).astype(np.float64)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation_9"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 10
    input_val = np.random.randn(1, 10, 10, 1).astype(np.float32)
    filter_val = np.random.randn(3, 3, 1).astype(np.float32)
    strides = [1, 2, 2, 1]
    rates = [1, 2, 2, 1]
    padding = "VALID"
    name = "dilation_10"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Dilation2D"] = tf_raw_ops_Dilation2D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Dilation2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Dilation2D'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Dilation2D', generated_inputs['tf.raw_ops.Dilation2D'], lib="tf", suffix=0)
