
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FusedBatchNormGrad_inputs():
    list_of_inputs = []

    # Input 1
    y_backprop = np.random.rand(1, 32, 32, 3).astype(np.float32)
    x = np.random.rand(1, 32, 32, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    reserve_space_1 = np.random.rand(3).astype(np.float32)
    reserve_space_2 = np.random.rand(3).astype(np.float32)
    epsilon = 0.001
    data_format = "NHWC"
    is_training = True
    name = "batch_norm_grad_1"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    y_backprop = np.random.rand(1, 3, 32, 32).astype(np.float32)
    x = np.random.rand(1, 3, 32, 32).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    reserve_space_1 = np.random.rand(3).astype(np.float32)
    reserve_space_2 = np.random.rand(3).astype(np.float32)
    epsilon = 0.00001
    data_format = "NCHW"
    is_training = False
    name = "batch_norm_grad_2"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    y_backprop = np.random.rand(2, 64, 64, 5).astype(np.float32)
    x = np.random.rand(2, 64, 64, 5).astype(np.float32)
    scale = np.random.rand(5).astype(np.float32)
    reserve_space_1 = np.random.rand(5).astype(np.float32)
    reserve_space_2 = np.random.rand(5).astype(np.float32)
    epsilon = 0.1
    data_format = "NHWC"
    is_training = True
    name = "batch_norm_grad_3"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4
    y_backprop = np.random.rand(4, 16, 16, 2).astype(np.float32)
    x = np.random.rand(4, 16, 16, 2).astype(np.float32)
    scale = np.random.rand(2).astype(np.float32)
    reserve_space_1 = np.random.rand(2).astype(np.float32)
    reserve_space_2 = np.random.rand(2).astype(np.float32)
    epsilon = 0.0001
    data_format = "NHWC"
    is_training = False
    name = "batch_norm_grad_4"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    y_backprop = np.random.rand(1, 8, 8, 1).astype(np.float32)
    x = np.random.rand(1, 8, 8, 1).astype(np.float32)
    scale = np.random.rand(1).astype(np.float32)
    reserve_space_1 = np.random.rand(1).astype(np.float32)
    reserve_space_2 = np.random.rand(1).astype(np.float32)
    epsilon = 1e-8
    data_format = "NHWC"
    is_training = True
    name = "batch_norm_grad_5"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    y_backprop = np.random.rand(2, 4, 4, 3).astype(np.float32)
    x = np.random.rand(2, 4, 4, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    reserve_space_1 = np.random.rand(3).astype(np.float32)
    reserve_space_2 = np.random.rand(3).astype(np.float32)
    epsilon = 0.5
    data_format = "NHWC"
    is_training = False
    name = "batch_norm_grad_6"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    y_backprop = np.random.rand(1, 3, 64, 64).astype(np.float32)
    x = np.random.rand(1, 3, 64, 64).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    reserve_space_1 = np.random.rand(3).astype(np.float32)
    reserve_space_2 = np.random.rand(3).astype(np.float32)
    epsilon = 0.0001
    data_format = "NCHW"
    is_training = True
    name = "batch_norm_grad_7"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    y_backprop = np.random.rand(1, 6, 128, 128).astype(np.float32)
    x = np.random.rand(1, 6, 128, 128).astype(np.float32)
    scale = np.random.rand(6).astype(np.float32)
    reserve_space_1 = np.random.rand(6).astype(np.float32)
    reserve_space_2 = np.random.rand(6).astype(np.float32)
    epsilon = 0.01
    data_format = "NCHW"
    is_training = False
    name = "batch_norm_grad_8"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    y_backprop = np.random.rand(2, 2, 2, 2).astype(np.float32)
    x = np.random.rand(2, 2, 2, 2).astype(np.float32)
    scale = np.random.rand(2).astype(np.float32)
    reserve_space_1 = np.random.rand(2).astype(np.float32)
    reserve_space_2 = np.random.rand(2).astype(np.float32)
    epsilon = 0.0001
    data_format = "NHWC"
    is_training = True
    name = "batch_norm_grad_9"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    y_backprop = np.random.rand(4, 8, 8, 4).astype(np.float32)
    x = np.random.rand(4, 8, 8, 4).astype(np.float32)
    scale = np.random.rand(4).astype(np.float32)
    reserve_space_1 = np.random.rand(4).astype(np.float32)
    reserve_space_2 = np.random.rand(4).astype(np.float32)
    epsilon = 0.0000001
    data_format = "NHWC"
    is_training = False
    name = "batch_norm_grad_10"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FusedBatchNormGrad"] = tf_raw_ops_FusedBatchNormGrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FusedBatchNormGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedBatchNormGrad'.")

check_valid('tf.raw_ops.FusedBatchNormGrad', generated_inputs['tf.raw_ops.FusedBatchNormGrad'], lib="tf", suffix=0)
