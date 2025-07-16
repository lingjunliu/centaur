
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fused_batch_norm_grad_v3_inputs():
    list_of_inputs = []

    # Input 1: Basic NHWC, is_training=True
    y_backprop = np.random.randn(1, 5, 5, 3).astype(np.float32)
    x = np.random.randn(1, 5, 5, 3).astype(np.float32)
    scale = np.random.randn(3).astype(np.float32)
    reserve_space_1 = np.random.randn(3).astype(np.float32)
    reserve_space_2 = np.random.randn(3).astype(np.float32)
    reserve_space_3 = np.random.randn(3).astype(np.float32)

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "reserve_space_3": reserve_space_3,
        "epsilon": 0.0001,
        "data_format": "NHWC",
        "is_training": True,
        "name": "basic_nhwc_training"
    }
    list_of_inputs.append(input_dict)

    # Input 2: Basic NCHW, is_training=True
    y_backprop = np.random.randn(1, 3, 5, 5).astype(np.float32)
    x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    scale = np.random.randn(3).astype(np.float32)
    reserve_space_1 = np.random.randn(3).astype(np.float32)
    reserve_space_2 = np.random.randn(3).astype(np.float32)
    reserve_space_3 = np.random.randn(3).astype(np.float32)

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "reserve_space_3": reserve_space_3,
        "epsilon": 0.0001,
        "data_format": "NCHW",
        "is_training": True,
        "name": "basic_nchw_training"
    }
    list_of_inputs.append(input_dict)

    # Input 3: Basic NHWC, is_training=False
    y_backprop = np.random.randn(1, 5, 5, 3).astype(np.float32)
    x = np.random.randn(1, 5, 5, 3).astype(np.float32)
    scale = np.random.randn(3).astype(np.float32)
    reserve_space_1 = np.random.randn(3).astype(np.float32)
    reserve_space_2 = np.random.randn(3).astype(np.float32)
    reserve_space_3 = np.random.randn(3).astype(np.float32)

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "reserve_space_3": reserve_space_3,
        "epsilon": 0.0001,
        "data_format": "NHWC",
        "is_training": False,
        "name": "basic_nhwc_inference"
    }
    list_of_inputs.append(input_dict)

    # Input 4: Different epsilon value
    y_backprop = np.random.randn(1, 5, 5, 3).astype(np.float32)
    x = np.random.randn(1, 5, 5, 3).astype(np.float32)
    scale = np.random.randn(3).astype(np.float32)
    reserve_space_1 = np.random.randn(3).astype(np.float32)
    reserve_space_2 = np.random.randn(3).astype(np.float32)
    reserve_space_3 = np.random.randn(3).astype(np.float32)

    input_dict = {
        "y_backprop": y_backprop,
        "x": x,
        "scale": scale,
        "reserve_space_1": reserve_space_1,
        "reserve_space_2": reserve_space_2,
        "reserve_space_3": reserve_space_3,
        "epsilon": 0.001,
        "data_format": "NHWC",
        "is_training": True,
        "name": "different_epsilon"
    }
    list_of_inputs.append(input_dict)

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
