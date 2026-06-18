
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_dilation2d_backprop_filter_inputs():
    list_of_inputs = []
    
    # Case 1: Float32, strides=[1,1,1,1], rates=[1,1,1,1], VALID padding
    input_val = np.random.randn(1, 3, 3, 1).astype(np.float32)
    filter_val = np.random.randn(2, 2, 1).astype(np.float32)
    out_backprop_val = np.random.randn(1, 2, 2, 1).astype(np.float32)
    list_of_inputs.append({
        'name': 'case1',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 1, 1, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'VALID'
    })

    # Case 2: Float64, strides=[1,1,1,1], rates=[1,1,1,1], SAME padding
    input_val = np.random.randn(1, 3, 3, 1).astype(np.float64)
    filter_val = np.random.randn(2, 2, 1).astype(np.float64)
    out_backprop_val = np.random.randn(1, 3, 3, 1).astype(np.float64)
    list_of_inputs.append({
        'name': 'case2',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 1, 1, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'SAME'
    })

    # Case 3: Int32, strides=[1,2,2,1], rates=[1,1,1,1], VALID padding
    input_val = np.random.randint(-10, 10, size=(2, 5, 5, 2)).astype(np.int32)
    filter_val = np.random.randint(-10, 10, size=(3, 3, 2)).astype(np.int32)
    out_backprop_val = np.random.randint(-10, 10, size=(2, 2, 2, 2)).astype(np.int32)
    list_of_inputs.append({
        'name': 'case3',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 2, 2, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'VALID'
    })

    # Case 4: Float32 with larger batch/channel, strides=[1,2,2,1], rates=[1,1,1,1], SAME padding
    input_val = np.random.randn(2, 5, 5, 2).astype(np.float32)
    filter_val = np.random.randn(3, 3, 2).astype(np.float32)
    out_backprop_val = np.random.randn(2, 3, 3, 2).astype(np.float32)
    list_of_inputs.append({
        'name': 'case4',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 2, 2, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'SAME'
    })

    # Case 5: Float32, strides=[1,1,1,1], rates=[1,2,2,1], VALID padding
    input_val = np.random.randn(1, 5, 5, 1).astype(np.float32)
    filter_val = np.random.randn(2, 2, 1).astype(np.float32)
    out_backprop_val = np.random.randn(1, 3, 3, 1).astype(np.float32)
    list_of_inputs.append({
        'name': 'case5',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 1, 1, 1],
        'rates': [1, 2, 2, 1],
        'padding': 'VALID'
    })

    # Case 6: Float64, strides=[1,1,1,1], rates=[1,2,2,1], SAME padding
    input_val = np.random.randn(1, 5, 5, 1).astype(np.float64)
    filter_val = np.random.randn(2, 2, 1).astype(np.float64)
    out_backprop_val = np.random.randn(1, 5, 5, 1).astype(np.float64)
    list_of_inputs.append({
        'name': 'case6',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 1, 1, 1],
        'rates': [1, 2, 2, 1],
        'padding': 'SAME'
    })

    # Case 7: Float32 with 3 channels, strides=[1,1,1,1], rates=[1,1,1,1], VALID padding
    input_val = np.random.randn(4, 4, 4, 3).astype(np.float32)
    filter_val = np.random.randn(2, 2, 3).astype(np.float32)
    out_backprop_val = np.random.randn(4, 3, 3, 3).astype(np.float32)
    list_of_inputs.append({
        'name': 'case7',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 1, 1, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'VALID'
    })

    # Case 8: Int64, strides=[1,1,2,1], rates=[1,1,1,1], SAME padding
    input_val = np.random.randint(-100, 100, size=(2, 4, 4, 2)).astype(np.int64)
    filter_val = np.random.randint(-100, 100, size=(2, 2, 2)).astype(np.int64)
    out_backprop_val = np.random.randint(-100, 100, size=(2, 4, 2, 2)).astype(np.int64)
    list_of_inputs.append({
        'name': 'case8',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 1, 2, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'SAME'
    })

    # Case 9: Float32, strides=[1,3,3,1], rates=[1,1,1,1], VALID padding
    input_val = np.random.randn(1, 7, 7, 1).astype(np.float32)
    filter_val = np.random.randn(3, 3, 1).astype(np.float32)
    out_backprop_val = np.random.randn(1, 2, 2, 1).astype(np.float32)
    list_of_inputs.append({
        'name': 'case9',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 3, 3, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'VALID'
    })

    # Case 10: Float32 with unequal stride and rate, SAME padding
    input_val = np.random.randn(2, 6, 4, 2).astype(np.float32)
    filter_val = np.random.randn(2, 3, 2).astype(np.float32)
    out_backprop_val = np.random.randn(2, 2, 4, 2).astype(np.float32)
    list_of_inputs.append({
        'name': 'case10',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 3, 1, 1],
        'rates': [1, 2, 1, 1],
        'padding': 'SAME'
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Dilation2DBackpropFilter"] = tf_dilation2d_backprop_filter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Dilation2DBackpropFilter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Dilation2DBackpropFilter'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Dilation2DBackpropFilter', generated_inputs['tf.raw_ops.Dilation2DBackpropFilter'], lib="tf", suffix=0)
