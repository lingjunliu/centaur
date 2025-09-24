
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FusedBatchNormGradV2_inputs():
    list_of_inputs = []

    # Input 1
    y_backprop = np.random.randn(1, 28, 28, 3).astype(np.float32)
    x = np.random.randn(1, 28, 28, 3).astype(np.float32)
    scale = np.random.randn(3).astype(np.float32)
    reserve_space_1 = np.random.randn(3).astype(np.float32)
    reserve_space_2 = np.random.randn(3).astype(np.float32)
    epsilon = 0.001
    data_format = 'NHWC'
    is_training = True
    name = 'bn_grad_1'

    input_dict = {
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name,
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    y_backprop = np.random.randn(1, 3, 28, 28).astype(np.float32)
    x = np.random.randn(1, 3, 28, 28).astype(np.float32)
    scale = np.random.randn(3).astype(np.float32)
    reserve_space_1 = np.random.randn(3).astype(np.float32)
    reserve_space_2 = np.random.randn(3).astype(np.float32)
    epsilon = 0.00001
    data_format = 'NCHW'
    is_training = False
    name = 'bn_grad_2'

    input_dict = {
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name,
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    y_backprop = np.random.randn(2, 14, 14, 5).astype(np.float32)
    x = np.random.randn(2, 14, 14, 5).astype(np.float32)
    scale = np.random.randn(5).astype(np.float32)
    reserve_space_1 = np.random.randn(5).astype(np.float32)
    reserve_space_2 = np.random.randn(5).astype(np.float32)
    epsilon = 0.1
    data_format = 'NHWC'
    is_training = True
    name = 'bn_grad_3'

    input_dict = {
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name,
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    y_backprop = np.random.randn(4, 4, 4, 4).astype(np.float32)
    x = np.random.randn(4, 4, 4, 4).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)
    reserve_space_1 = np.random.randn(4).astype(np.float32)
    reserve_space_2 = np.random.randn(4).astype(np.float32)
    epsilon = 0.0001
    data_format = 'NHWC'
    is_training = False
    name = 'bn_grad_5'

    input_dict = {
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name,
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    y_backprop = np.random.randn(1, 64, 64, 1).astype(np.float32)
    x = np.random.randn(1, 64, 64, 1).astype(np.float32)
    scale = np.random.randn(1).astype(np.float32)
    reserve_space_1 = np.random.randn(1).astype(np.float32)
    reserve_space_2 = np.random.randn(1).astype(np.float32)
    epsilon = 0.0001
    data_format = 'NHWC'
    is_training = True
    name = 'bn_grad_6'

    input_dict = {
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name,
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    y_backprop = np.random.randn(1, 1, 32, 32).astype(np.float32)
    x = np.random.randn(1, 1, 32, 32).astype(np.float32)
    scale = np.random.randn(1).astype(np.float32)
    reserve_space_1 = np.random.randn(1).astype(np.float32)
    reserve_space_2 = np.random.randn(1).astype(np.float32)
    epsilon = 0.01
    data_format = 'NCHW'
    is_training = True
    name = 'bn_grad_7'

    input_dict = {
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name,
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    y_backprop = np.random.randn(2, 3, 16, 16).astype(np.float32)
    x = np.random.randn(2, 3, 16, 16).astype(np.float32)
    scale = np.random.randn(3).astype(np.float32)
    reserve_space_1 = np.random.randn(3).astype(np.float32)
    reserve_space_2 = np.random.randn(3).astype(np.float32)
    epsilon = 0.0001
    data_format = 'NCHW'
    is_training = False
    name = 'bn_grad_8'

    input_dict = {
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name,
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    y_backprop = np.random.randn(1, 1, 1, 1).astype(np.float32)
    x = np.random.randn(1, 1, 1, 1).astype(np.float32)
    scale = np.random.randn(1).astype(np.float32)
    reserve_space_1 = np.random.randn(1).astype(np.float32)
    reserve_space_2 = np.random.randn(1).astype(np.float32)
    epsilon = 1e-8
    data_format = 'NHWC'
    is_training = True
    name = 'bn_grad_9'

    input_dict = {
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name,
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    y_backprop = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    scale = np.random.randn(3).astype(np.float32)
    reserve_space_1 = np.random.randn(3).astype(np.float32)
    reserve_space_2 = np.random.randn(3).astype(np.float32)
    epsilon = 0.5
    data_format = 'NCHW'
    is_training = True
    name = 'bn_grad_10'

    input_dict = {
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name,
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: half
    y_backprop = np.random.randn(1, 28, 28, 3).astype(np.float16)
    x = np.random.randn(1, 28, 28, 3).astype(np.float16)
    scale = np.random.randn(3).astype(np.float32)
    reserve_space_1 = np.random.randn(3).astype(np.float32)
    reserve_space_2 = np.random.randn(3).astype(np.float32)
    epsilon = 0.001
    data_format = 'NHWC'
    is_training = True
    name = 'bn_grad_11'

    input_dict = {
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name,
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FusedBatchNormGradV2"] = tf_raw_ops_FusedBatchNormGradV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FusedBatchNormGradV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedBatchNormGradV2'.")

check_valid('tf.raw_ops.FusedBatchNormGradV2', generated_inputs['tf.raw_ops.FusedBatchNormGradV2'], lib="tf", suffix=0)
