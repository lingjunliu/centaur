
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fused_batch_norm_grad_v3_inputs():
    list_of_inputs = []

    # Input 1
    y_backprop = np.random.rand(1, 28, 28, 3).astype(np.float32)
    x = np.random.rand(1, 28, 28, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    reserve_space_1 = np.random.rand(3).astype(np.float32)
    reserve_space_2 = np.random.rand(3).astype(np.float32)
    reserve_space_3 = np.random.rand(3).astype(np.float32)
    epsilon = 0.001
    data_format = "NHWC"
    is_training = True
    name = "batchnorm_grad_1"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "reserve_space_3": reserve_space_3,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    y_backprop = np.random.rand(2, 3, 4, 5).astype(np.float32)
    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    scale = np.random.rand(5).astype(np.float32)
    reserve_space_1 = np.random.rand(5).astype(np.float32)
    reserve_space_2 = np.random.rand(5).astype(np.float32)
    reserve_space_3 = np.random.rand(5).astype(np.float32)
    epsilon = 0.01
    data_format = "NHWC"
    is_training = False
    name = "batchnorm_grad_2"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "reserve_space_3": reserve_space_3,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 3
    y_backprop = np.random.rand(1, 3, 28, 28).astype(np.float32)
    x = np.random.rand(1, 3, 28, 28).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    reserve_space_1 = np.random.rand(3).astype(np.float32)
    reserve_space_2 = np.random.rand(3).astype(np.float32)
    reserve_space_3 = np.random.rand(3).astype(np.float32)
    epsilon = 0.0001
    data_format = "NCHW"
    is_training = True
    name = "batchnorm_grad_3"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "reserve_space_3": reserve_space_3,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    y_backprop = np.random.rand(4, 5, 6, 7).astype(np.float32)
    x = np.random.rand(4, 5, 6, 7).astype(np.float32)
    scale = np.random.rand(7).astype(np.float32)
    reserve_space_1 = np.random.rand(7).astype(np.float32)
    reserve_space_2 = np.random.rand(7).astype(np.float32)
    reserve_space_3 = np.random.rand(7).astype(np.float32)
    epsilon = 0.00001
    data_format = "NHWC"
    is_training = False
    name = "batchnorm_grad_4"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "reserve_space_3": reserve_space_3,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Using bfloat16
    y_backprop = np.random.rand(1, 28, 28, 3).astype(np.float16)
    x = np.random.rand(1, 28, 28, 3).astype(np.float16)
    scale = np.random.rand(3).astype(np.float32)
    reserve_space_1 = np.random.rand(3).astype(np.float32)
    reserve_space_2 = np.random.rand(3).astype(np.float32)
    reserve_space_3 = np.random.rand(3).astype(np.float32)
    epsilon = 0.001
    data_format = "NHWC"
    is_training = True
    name = "batchnorm_grad_5"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "reserve_space_3": reserve_space_3,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Using half
    y_backprop = np.random.rand(1, 28, 28, 3).astype(np.float16)
    x = np.random.rand(1, 28, 28, 3).astype(np.float16)
    scale = np.random.rand(3).astype(np.float32)
    reserve_space_1 = np.random.rand(3).astype(np.float32)
    reserve_space_2 = np.random.rand(3).astype(np.float32)
    reserve_space_3 = np.random.rand(3).astype(np.float32)
    epsilon = 0.001
    data_format = "NHWC"
    is_training = True
    name = "batchnorm_grad_6"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "reserve_space_3": reserve_space_3,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NDHWC format and training=True to avoid error.
    y_backprop = np.random.rand(2, 3, 4, 5, 6).astype(np.float32)
    x = np.random.rand(2, 3, 4, 5, 6).astype(np.float32)
    scale = np.random.rand(6).astype(np.float32)
    reserve_space_1 = np.random.rand(6).astype(np.float32)
    reserve_space_2 = np.random.rand(6).astype(np.float32)
    reserve_space_3 = np.random.rand(6).astype(np.float32)
    epsilon = 0.01
    data_format = "NDHWC"
    is_training = True
    name = "batchnorm_grad_7"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "reserve_space_3": reserve_space_3,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: NCDHW format and training=True to avoid error.
    y_backprop = np.random.rand(2, 6, 4, 5, 3).astype(np.float32)
    x = np.random.rand(2, 6, 4, 5, 3).astype(np.float32)
    scale = np.random.rand(6).astype(np.float32)
    reserve_space_1 = np.random.rand(6).astype(np.float32)
    reserve_space_2 = np.random.rand(6).astype(np.float32)
    reserve_space_3 = np.random.rand(6).astype(np.float32)
    epsilon = 0.01
    data_format = "NCDHW"
    is_training = True
    name = "batchnorm_grad_8"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "reserve_space_3": reserve_space_3,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: epsilon close to zero
    y_backprop = np.random.rand(1, 28, 28, 3).astype(np.float32)
    x = np.random.rand(1, 28, 28, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    reserve_space_1 = np.random.rand(3).astype(np.float32)
    reserve_space_2 = np.random.rand(3).astype(np.float32)
    reserve_space_3 = np.random.rand(3).astype(np.float32)
    epsilon = 1e-8
    data_format = "NHWC"
    is_training = True
    name = "batchnorm_grad_9"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "reserve_space_3": reserve_space_3,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Different shapes
    y_backprop = np.random.rand(2, 14, 14, 6).astype(np.float32)
    x = np.random.rand(2, 14, 14, 6).astype(np.float32)
    scale = np.random.rand(6).astype(np.float32)
    reserve_space_1 = np.random.rand(6).astype(np.float32)
    reserve_space_2 = np.random.rand(6).astype(np.float32)
    reserve_space_3 = np.random.rand(6).astype(np.float32)
    epsilon = 0.001
    data_format = "NHWC"
    is_training = True
    name = "batchnorm_grad_10"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "reserve_space_3": reserve_space_3,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: NHWC with training=False, correct channel size
    y_backprop = np.random.rand(2, 3, 4, 3).astype(np.float32)
    x = np.random.rand(2, 3, 4, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    reserve_space_1 = np.random.rand(3).astype(np.float32)
    reserve_space_2 = np.random.rand(3).astype(np.float32)
    reserve_space_3 = np.random.rand(3).astype(np.float32)
    epsilon = 0.01
    data_format = "NHWC"
    is_training = False
    name = "batchnorm_grad_11"

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "reserve_space_3": reserve_space_3,
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FusedBatchNormGradV3"] = tf_raw_ops_fused_batch_norm_grad_v3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FusedBatchNormGradV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedBatchNormGradV3'.")

check_valid('tf.raw_ops.FusedBatchNormGradV3', generated_inputs['tf.raw_ops.FusedBatchNormGradV3'], lib="tf", suffix=0)
