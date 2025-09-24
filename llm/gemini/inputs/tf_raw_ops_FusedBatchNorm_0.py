
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fused_batch_norm_inputs():
    list_of_inputs = []

    # Input 1
    x = np.random.rand(1, 5, 5, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 1.0
    data_format = "NHWC"
    is_training = True
    name = "batch_norm_1"

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
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.random.rand(2, 10, 10, 1).astype(np.float32)
    scale = np.random.rand(1).astype(np.float32)
    offset = np.random.rand(1).astype(np.float32)
    mean = np.random.rand(1).astype(np.float32)
    variance = np.random.rand(1).astype(np.float32)
    epsilon = 0.00001
    exponential_avg_factor = 0.5
    data_format = "NHWC"
    is_training = False
    name = "batch_norm_2"

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
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.random.rand(1, 3, 32, 32).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)
    epsilon = 0.1
    exponential_avg_factor = 0.9
    data_format = "NCHW"
    is_training = True
    name = "batch_norm_3"

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
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.random.rand(4, 8, 8, 4).astype(np.float32)
    scale = np.random.rand(4).astype(np.float32)
    offset = np.random.rand(4).astype(np.float32)
    mean = np.random.rand(4).astype(np.float32)
    variance = np.random.rand(4).astype(np.float32)
    epsilon = 0.0005
    exponential_avg_factor = 0.75
    data_format = "NHWC"
    is_training = False
    name = "batch_norm_4"

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
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.random.rand(1, 1, 1, 1).astype(np.float32)
    scale = np.random.rand(1).astype(np.float32)
    offset = np.random.rand(1).astype(np.float32)
    mean = np.random.rand(1).astype(np.float32)
    variance = np.random.rand(1).astype(np.float32)
    epsilon = 0.01
    exponential_avg_factor = 0.25
    data_format = "NHWC"
    is_training = True
    name = "batch_norm_5"

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
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    x = np.random.rand(2, 4, 4, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)
    epsilon = 0.002
    exponential_avg_factor = 0.8
    data_format = "NHWC"
    is_training = True
    name = "batch_norm_6"

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
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    x = np.random.rand(1, 2, 2, 5).astype(np.float32)
    scale = np.random.rand(5).astype(np.float32)
    offset = np.random.rand(5).astype(np.float32)
    mean = np.random.rand(5).astype(np.float32)
    variance = np.random.rand(5).astype(np.float32)
    epsilon = 0.00005
    exponential_avg_factor = 0.6
    data_format = "NHWC"
    is_training = False
    name = "batch_norm_7"

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
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.random.rand(1, 16, 16, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)
    epsilon = 0.0002
    exponential_avg_factor = 0.3
    data_format = "NHWC"
    is_training = True
    name = "batch_norm_8"

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
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.random.rand(1, 4, 28, 28).astype(np.float32)
    scale = np.random.rand(4).astype(np.float32)
    offset = np.random.rand(4).astype(np.float32)
    mean = np.random.rand(4).astype(np.float32)
    variance = np.random.rand(4).astype(np.float32)
    epsilon = 0.000001
    exponential_avg_factor = 0.1
    data_format = "NCHW"
    is_training = False
    name = "batch_norm_9"

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
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.random.rand(1, 64, 64, 1).astype(np.float32)
    scale = np.random.rand(1).astype(np.float32)
    offset = np.random.rand(1).astype(np.float32)
    mean = np.random.rand(1).astype(np.float32)
    variance = np.random.rand(1).astype(np.float32)
    epsilon = 0.0000001
    exponential_avg_factor = 0.05
    data_format = "NHWC"
    is_training = True
    name = "batch_norm_10"

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
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FusedBatchNorm"] = tf_raw_ops_fused_batch_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FusedBatchNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedBatchNorm'.")

check_valid('tf.raw_ops.FusedBatchNorm', generated_inputs['tf.raw_ops.FusedBatchNorm'], lib="tf", suffix=0)
