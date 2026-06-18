
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FusedResizeAndPadConv2D_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'resize_align_corners': False,
        'name': "op1",
        'input': np.random.rand(2, 8, 8, 3).astype(np.float32),
        'size': np.array([4, 4], dtype=np.int32),
        'paddings': np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(3, 3, 3, 2).astype(np.float32),
        'mode': "REFLECT",
        'strides': [1, 1, 1, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'resize_align_corners': True,
        'name': "op2",
        'input': np.random.rand(1, 10, 10, 1).astype(np.float32),
        'size': np.array([5, 5], dtype=np.int32),
        'paddings': np.array([[0, 0], [2, 2], [2, 2], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(2, 2, 1, 4).astype(np.float32),
        'mode': "SYMMETRIC",
        'strides': [1, 2, 2, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'resize_align_corners': False,
        'name': "op3",
        'input': np.random.rand(3, 6, 6, 2).astype(np.float64),
        'size': np.array([3, 3], dtype=np.int32),
        'paddings': np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(2, 2, 2, 1).astype(np.float64),
        'mode': "REFLECT",
        'strides': [1, 1, 1, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'resize_align_corners': True,
        'name': "op4",
        'input': np.random.rand(1, 12, 12, 4).astype(np.float16),
        'size': np.array([6, 6], dtype=np.int32),
        'paddings': np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(3, 3, 4, 2).astype(np.float16),
        'mode': "REFLECT",
        'strides': [1, 1, 1, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'resize_align_corners': False,
        'name': "op5",
        'input': np.random.rand(2, 16, 16, 3).astype(np.float32),
        'size': np.array([8, 8], dtype=np.int32),
        'paddings': np.array([[0, 0], [2, 1], [1, 2], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(3, 3, 3, 8).astype(np.float32),
        'mode': "SYMMETRIC",
        'strides': [1, 1, 1, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'resize_align_corners': True,
        'name': "op6",
        'input': np.random.rand(1, 5, 5, 2).astype(np.float32),
        'size': np.array([10, 10], dtype=np.int32),
        'paddings': np.array([[0, 0], [2, 2], [2, 2], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(5, 5, 2, 2).astype(np.float32),
        'mode': "REFLECT",
        'strides': [1, 1, 1, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'resize_align_corners': False,
        'name': "op7",
        'input': np.random.rand(4, 8, 8, 1).astype(np.float64),
        'size': np.array([2, 2], dtype=np.int32),
        'paddings': np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(1, 1, 1, 1).astype(np.float64),
        'mode': "REFLECT",
        'strides': [1, 1, 1, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'resize_align_corners': False,
        'name': "op8",
        'input': np.random.rand(1, 4, 4, 3).astype(np.float32),
        'size': np.array([3, 5], dtype=np.int32),
        'paddings': np.array([[0, 0], [1, 1], [2, 2], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(3, 3, 3, 4).astype(np.float32),
        'mode': "SYMMETRIC",
        'strides': [1, 1, 1, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'resize_align_corners': True,
        'name': "op9",
        'input': np.random.rand(2, 14, 14, 2).astype(np.float16),
        'size': np.array([7, 7], dtype=np.int32),
        'paddings': np.array([[0, 0], [3, 3], [3, 3], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(4, 4, 2, 4).astype(np.float16),
        'mode': "SYMMETRIC",
        'strides': [1, 2, 2, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'resize_align_corners': True,
        'name': "op10",
        'input': np.random.rand(1, 20, 20, 1).astype(np.float32),
        'size': np.array([5, 10], dtype=np.int32),
        'paddings': np.array([[0, 0], [1, 2], [3, 4], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(2, 2, 1, 1).astype(np.float32),
        'mode': "REFLECT",
        'strides': [1, 1, 1, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FusedResizeAndPadConv2D"] = tf_raw_ops_FusedResizeAndPadConv2D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FusedResizeAndPadConv2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedResizeAndPadConv2D'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.FusedResizeAndPadConv2D', generated_inputs['tf.raw_ops.FusedResizeAndPadConv2D'], lib="tf", suffix=0)
