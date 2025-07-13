
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fused_batch_norm_v2_inputs():
    list_of_inputs = []

    # Input 1
    x = np.random.rand(1, 28, 28, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 0.5
    data_format = 'NHWC'
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
        "name": "batch_norm_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.random.rand(1, 3, 28, 28).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)
    epsilon = 0.00001
    exponential_avg_factor = 1.0
    data_format = 'NCHW'
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
        "name": "batch_norm_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.random.rand(4, 16, 16, 32).astype(np.float32)
    scale = np.random.rand(32).astype(np.float32)
    offset = np.random.rand(32).astype(np.float32)
    mean = np.random.rand(32).astype(np.float32)
    variance = np.random.rand(32).astype(np.float32)
    epsilon = 0.0005
    exponential_avg_factor = 0.9
    data_format = 'NHWC'
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
        "name": "batch_norm_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    x = np.random.rand(4, 32, 16, 16).astype(np.float32)
    scale = np.random.rand(32).astype(np.float32)
    offset = np.random.rand(32).astype(np.float32)
    mean = np.random.rand(32).astype(np.float32)
    variance = np.random.rand(32).astype(np.float32)
    epsilon = 0.0005
    exponential_avg_factor = 0.9
    data_format = 'NCHW'
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
        "name": "batch_norm_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (half) - removed bfloat16 since numpy doesn't support it directly
    x = np.random.rand(1, 28, 28, 3).astype(np.float16)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 0.5
    data_format = 'NHWC'
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
        "name": "batch_norm_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.random.rand(2, 10, 10, 5).astype(np.float32)
    scale = np.random.rand(5).astype(np.float32)
    offset = np.random.rand(5).astype(np.float32)
    mean = np.random.rand(5).astype(np.float32)
    variance = np.random.rand(5).astype(np.float32)
    epsilon = 0.00005
    exponential_avg_factor = 0.8
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
        "name": "batch_norm_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.random.rand(2, 5, 10, 10).astype(np.float32)
    scale = np.random.rand(5).astype(np.float32)
    offset = np.random.rand(5).astype(np.float32)
    mean = np.random.rand(5).astype(np.float32)
    variance = np.random.rand(5).astype(np.float32)
    epsilon = 0.00005
    exponential_avg_factor = 0.8
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
        "name": "batch_norm_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.random.rand(8, 4, 4, 1).astype(np.float32)
    scale = np.random.rand(1).astype(np.float32)
    offset = np.random.rand(1).astype(np.float32)
    mean = np.random.rand(1).astype(np.float32)
    variance = np.random.rand(1).astype(np.float32)
    epsilon = 0.01
    exponential_avg_factor = 0.2
    data_format = 'NHWC'
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
        "name": "batch_norm_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.random.rand(8, 1, 4, 4).astype(np.float32)
    scale = np.random.rand(1).astype(np.float32)
    offset = np.random.rand(1).astype(np.float32)
    mean = np.random.rand(1).astype(np.float32)
    variance = np.random.rand(1).astype(np.float32)
    epsilon = 0.01
    exponential_avg_factor = 0.2
    data_format = 'NCHW'
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
        "name": "batch_norm_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.random.rand(1, 28, 28, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.zeros(3).astype(np.float32)
    variance = np.ones(3).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 0.5
    data_format = 'NHWC'
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
        "name": "batch_norm_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FusedBatchNormV2"] = tf_raw_ops_fused_batch_norm_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FusedBatchNormV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedBatchNormV2'.")

check_valid('tf.raw_ops.FusedBatchNormV2', generated_inputs['tf.raw_ops.FusedBatchNormV2'], lib="tf", suffix=0)
