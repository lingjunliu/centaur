
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_dilation2d_backprop_input_inputs():
    list_of_inputs = []

    # Input 1: float32, VALID, strides [1, 1, 1, 1]
    input_dict = {
        "input": np.random.randn(1, 5, 5, 1).astype(np.float32),
        "filter": np.random.randn(3, 3, 1).astype(np.float32),
        "out_backprop": np.random.randn(1, 3, 3, 1).astype(np.float32),
        "strides": [1, 1, 1, 1],
        "rates": [1, 1, 1, 1],
        "padding": "VALID",
        "name": "test_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, SAME, strides [1, 1, 1, 1]
    input_dict = {
        "input": np.random.randn(2, 4, 4, 3).astype(np.float64),
        "filter": np.random.randn(2, 2, 3).astype(np.float64),
        "out_backprop": np.random.randn(2, 4, 4, 3).astype(np.float64),
        "strides": [1, 1, 1, 1],
        "rates": [1, 1, 1, 1],
        "padding": "SAME",
        "name": "test_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32, VALID, strides [1, 2, 2, 1]
    input_dict = {
        "input": np.random.randint(-10, 10, size=(1, 6, 6, 2)).astype(np.int32),
        "filter": np.random.randint(-5, 5, size=(2, 2, 2)).astype(np.int32),
        "out_backprop": np.random.randint(-10, 10, size=(1, 3, 3, 2)).astype(np.int32),
        "strides": [1, 2, 2, 1],
        "rates": [1, 1, 1, 1],
        "padding": "VALID",
        "name": "test_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int32, SAME, strides [1, 2, 2, 1]
    input_dict = {
        "input": np.random.randint(-5, 5, size=(1, 5, 5, 1)).astype(np.int32),
        "filter": np.random.randint(-3, 3, size=(3, 3, 1)).astype(np.int32),
        "out_backprop": np.random.randint(-5, 5, size=(1, 3, 3, 1)).astype(np.int32),
        "strides": [1, 2, 2, 1],
        "rates": [1, 1, 1, 1],
        "padding": "SAME",
        "name": "test_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, VALID, strides [1, 1, 1, 1], rates [1, 2, 2, 1]
    input_dict = {
        "input": np.random.randn(1, 7, 7, 1).astype(np.float32),
        "filter": np.random.randn(3, 3, 1).astype(np.float32),
        "out_backprop": np.random.randn(1, 3, 3, 1).astype(np.float32),
        "strides": [1, 1, 1, 1],
        "rates": [1, 2, 2, 1],
        "padding": "VALID",
        "name": "test_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, SAME, strides [1, 1, 1, 1], rates [1, 2, 2, 1]
    input_dict = {
        "input": np.random.randn(1, 5, 5, 1).astype(np.float32),
        "filter": np.random.randn(3, 3, 1).astype(np.float32),
        "out_backprop": np.random.randn(1, 5, 5, 1).astype(np.float32),
        "strides": [1, 1, 1, 1],
        "rates": [1, 2, 2, 1],
        "padding": "SAME",
        "name": "test_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int32, VALID, strides [1, 1, 1, 1]
    input_dict = {
        "input": np.random.randint(-128, 127, size=(1, 10, 10, 4)).astype(np.int32),
        "filter": np.random.randint(-5, 5, size=(5, 5, 4)).astype(np.int32),
        "out_backprop": np.random.randint(-128, 127, size=(1, 6, 6, 4)).astype(np.int32),
        "strides": [1, 1, 1, 1],
        "rates": [1, 1, 1, 1],
        "padding": "VALID",
        "name": "test_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64, SAME, strides [1, 3, 3, 1]
    input_dict = {
        "input": np.random.randint(-1000, 1000, size=(1, 9, 9, 1)).astype(np.int64),
        "filter": np.random.randint(-100, 100, size=(3, 3, 1)).astype(np.int64),
        "out_backprop": np.random.randint(-1000, 1000, size=(1, 3, 3, 1)).astype(np.int64),
        "strides": [1, 3, 3, 1],
        "rates": [1, 1, 1, 1],
        "padding": "SAME",
        "name": "test_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, VALID, strides [1, 2, 2, 1], rates [1, 2, 2, 1]
    input_dict = {
        "input": np.random.randn(2, 10, 10, 2).astype(np.float64),
        "filter": np.random.randn(3, 3, 2).astype(np.float64),
        "out_backprop": np.random.randn(2, 3, 3, 2).astype(np.float64),
        "strides": [1, 2, 2, 1],
        "rates": [1, 2, 2, 1],
        "padding": "VALID",
        "name": "test_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, SAME, strides [1, 2, 2, 1], rates [1, 2, 2, 1]
    input_dict = {
        "input": np.random.randn(1, 6, 6, 1).astype(np.float32),
        "filter": np.random.randn(3, 3, 1).astype(np.float32),
        "out_backprop": np.random.randn(1, 3, 3, 1).astype(np.float32),
        "strides": [1, 2, 2, 1],
        "rates": [1, 2, 2, 1],
        "padding": "SAME",
        "name": "test_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Dilation2DBackpropInput"] = tf_dilation2d_backprop_input_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Dilation2DBackpropInput' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Dilation2DBackpropInput'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Dilation2DBackpropInput', generated_inputs['tf.raw_ops.Dilation2DBackpropInput'], lib="tf", suffix=0)
