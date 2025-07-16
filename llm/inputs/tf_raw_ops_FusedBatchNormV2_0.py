
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FusedBatchNormV2_inputs():
    list_of_inputs = []

    # Input 1: Basic training example with float32
    x = np.random.rand(1, 32, 32, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.array([0,0,0]).astype(np.float32)
    variance = np.array([1,1,1]).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 1.0
    data_format = "NHWC"
    is_training = True

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": epsilon,
        "exponential_avg_factor": exponential_avg_factor,
        "data_format": data_format,
        "is_training": is_training,
        "name": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Inference example, NCHW, float32
    x = np.random.rand(1, 3, 32, 32).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)
    epsilon = 0.00001
    exponential_avg_factor = 1.0
    data_format = "NCHW"
    is_training = False

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": epsilon,
        "exponential_avg_factor": exponential_avg_factor,
        "data_format": data_format,
        "is_training": is_training,
        "name": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Training, bfloat16
    x = np.random.rand(1, 32, 32, 3).astype(np.float16)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.array([0,0,0]).astype(np.float32)
    variance = np.array([1,1,1]).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 1.0
    data_format = "NHWC"
    is_training = True

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": epsilon,
        "exponential_avg_factor": exponential_avg_factor,
        "data_format": data_format,
        "is_training": is_training,
        "name": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Inference, half, NCHW
    x = np.random.rand(1, 3, 32, 32).astype(np.float16)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)
    epsilon = 0.00001
    exponential_avg_factor = 1.0
    data_format = "NCHW"
    is_training = False

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": epsilon,
        "exponential_avg_factor": exponential_avg_factor,
        "data_format": data_format,
        "is_training": is_training,
        "name": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Training with different shape
    x = np.random.rand(4, 16, 16, 1).astype(np.float32)
    scale = np.random.rand(1).astype(np.float32)
    offset = np.random.rand(1).astype(np.float32)
    mean = np.array([0]).astype(np.float32)
    variance = np.array([1]).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 1.0
    data_format = "NHWC"
    is_training = True

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": epsilon,
        "exponential_avg_factor": exponential_avg_factor,
        "data_format": data_format,
        "is_training": is_training,
        "name": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Inference with different shape
    x = np.random.rand(4, 1, 16, 16).astype(np.float32)
    scale = np.random.rand(1).astype(np.float32)
    offset = np.random.rand(1).astype(np.float32)
    mean = np.random.rand(1).astype(np.float32)
    variance = np.random.rand(1).astype(np.float32)
    epsilon = 0.00001
    exponential_avg_factor = 1.0
    data_format = "NCHW"
    is_training = False

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": epsilon,
        "exponential_avg_factor": exponential_avg_factor,
        "data_format": data_format,
        "is_training": is_training,
        "name": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Training with epsilon near zero
    x = np.random.rand(1, 32, 32, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.array([0,0,0]).astype(np.float32)
    variance = np.array([1,1,1]).astype(np.float32)
    epsilon = 1e-8
    exponential_avg_factor = 1.0
    data_format = "NHWC"
    is_training = True

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": epsilon,
        "exponential_avg_factor": exponential_avg_factor,
        "data_format": data_format,
        "is_training": is_training,
        "name": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Training with non-default exponential_avg_factor
    x = np.random.rand(1, 32, 32, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.array([0,0,0]).astype(np.float32)
    variance = np.array([1,1,1]).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 0.5
    data_format = "NHWC"
    is_training = True

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": epsilon,
        "exponential_avg_factor": exponential_avg_factor,
        "data_format": data_format,
        "is_training": is_training,
        "name": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Inference, different batch size
    x = np.random.rand(8, 3, 32, 32).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)
    epsilon = 0.00001
    exponential_avg_factor = 1.0
    data_format = "NCHW"
    is_training = False

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": epsilon,
        "exponential_avg_factor": exponential_avg_factor,
        "data_format": data_format,
        "is_training": is_training,
        "name": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Training bfloat16 different shape
    x = np.random.rand(2, 8, 8, 4).astype(np.float16)
    scale = np.random.rand(4).astype(np.float32)
    offset = np.random.rand(4).astype(np.float32)
    mean = np.array([0,0,0,0]).astype(np.float32)
    variance = np.array([1,1,1,1]).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 1.0
    data_format = "NHWC"
    is_training = True

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": epsilon,
        "exponential_avg_factor": exponential_avg_factor,
        "data_format": data_format,
        "is_training": is_training,
        "name": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    final_list = []
    for input_dict in list_of_inputs:
      final_list.append({"args": [], "kwargs": input_dict})
    return final_list

generated_inputs = {}
generated_inputs["tf.raw_ops.FusedBatchNormV2"] = tf_raw_ops_FusedBatchNormV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FusedBatchNormV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedBatchNormV2'.")

check_valid('tf.raw_ops.FusedBatchNormV2', generated_inputs['tf.raw_ops.FusedBatchNormV2'], lib="tf", suffix=0)
