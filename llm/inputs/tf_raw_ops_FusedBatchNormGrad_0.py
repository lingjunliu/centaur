
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fused_batch_norm_grad_inputs():
    list_of_inputs = []

    # Input 1
    y_backprop = np.random.rand(1, 28, 28, 3).astype(np.float32)
    x = np.random.rand(1, 28, 28, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    reserve_space_1 = np.random.rand(3).astype(np.float32)
    reserve_space_2 = np.random.rand(3).astype(np.float32)
    epsilon = 0.001
    data_format = "NHWC"
    is_training = True
    name = "batch_norm_grad_1"

    input_dict = {
        "y_backprop": tf.constant(y_backprop),
        "x": tf.constant(x),
        "scale": tf.constant(scale),
        "reserve_space_1": tf.constant(reserve_space_1),
        "reserve_space_2": tf.constant(reserve_space_2),
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 2
    y_backprop = np.random.rand(1, 3, 28, 28).astype(np.float32)
    x = np.random.rand(1, 3, 28, 28).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    reserve_space_1 = np.random.rand(3).astype(np.float32)
    reserve_space_2 = np.random.rand(3).astype(np.float32)
    epsilon = 0.00001
    data_format = "NCHW"
    is_training = False
    name = "batch_norm_grad_2"

    input_dict = {
        "y_backprop": tf.constant(y_backprop),
        "x": tf.constant(x),
        "scale": tf.constant(scale),
        "reserve_space_1": tf.constant(reserve_space_1),
        "reserve_space_2": tf.constant(reserve_space_2),
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 3
    y_backprop = np.random.rand(4, 14, 14, 8).astype(np.float32)
    x = np.random.rand(4, 14, 14, 8).astype(np.float32)
    scale = np.random.rand(8).astype(np.float32)
    reserve_space_1 = np.random.rand(8).astype(np.float32)
    reserve_space_2 = np.random.rand(8).astype(np.float32)
    epsilon = 0.1
    data_format = "NHWC"
    is_training = True
    name = "batch_norm_grad_3"

    input_dict = {
        "y_backprop": tf.constant(y_backprop),
        "x": tf.constant(x),
        "scale": tf.constant(scale),
        "reserve_space_1": tf.constant(reserve_space_1),
        "reserve_space_2": tf.constant(reserve_space_2),
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 4
    y_backprop = np.random.rand(2, 16, 16, 4).astype(np.float32)
    x = np.random.rand(2, 16, 16, 4).astype(np.float32)
    scale = np.random.rand(4).astype(np.float32)
    reserve_space_1 = np.random.rand(4).astype(np.float32)
    reserve_space_2 = np.random.rand(4).astype(np.float32)
    epsilon = 0.0001
    data_format = "NHWC"
    is_training = False
    name = "batch_norm_grad_4"

    input_dict = {
        "y_backprop": tf.constant(y_backprop),
        "x": tf.constant(x),
        "scale": tf.constant(scale),
        "reserve_space_1": tf.constant(reserve_space_1),
        "reserve_space_2": tf.constant(reserve_space_2),
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 5
    y_backprop = np.random.rand(1, 64, 64, 1).astype(np.float32)
    x = np.random.rand(1, 64, 64, 1).astype(np.float32)
    scale = np.random.rand(1).astype(np.float32)
    reserve_space_1 = np.random.rand(1).astype(np.float32)
    reserve_space_2 = np.random.rand(1).astype(np.float32)
    epsilon = 0.01
    data_format = "NHWC"
    is_training = True
    name = "batch_norm_grad_5"

    input_dict = {
        "y_backprop": tf.constant(y_backprop),
        "x": tf.constant(x),
        "scale": tf.constant(scale),
        "reserve_space_1": tf.constant(reserve_space_1),
        "reserve_space_2": tf.constant(reserve_space_2),
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FusedBatchNormGrad"] = tf_raw_ops_fused_batch_norm_grad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FusedBatchNormGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedBatchNormGrad'.")

check_valid('tf.raw_ops.FusedBatchNormGrad', generated_inputs['tf.raw_ops.FusedBatchNormGrad'], lib="tf", suffix=0)
