
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fused_batch_norm_v2_inputs():
    list_of_inputs = []

    # Input 1: Basic training case with NHWC
    x = np.random.rand(1, 28, 28, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.zeros(3).astype(np.float32)
    variance = np.ones(3).astype(np.float32)
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

    # Input 2: Basic training case with NCHW
    x = np.random.rand(1, 3, 28, 28).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.zeros(3).astype(np.float32)
    variance = np.ones(3).astype(np.float32)
    epsilon = 0.00001
    exponential_avg_factor = 1.0
    data_format = "NCHW"
    is_training = True
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

    # Input 3: Inference case with NHWC
    x = np.random.rand(1, 28, 28, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)
    epsilon = 0.0001
    exponential_avg_factor = 1
    data_format = "NHWC"
    is_training = False
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

    # Input 4: Inference case with NCHW
    x = np.random.rand(1, 3, 28, 28).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)
    epsilon = 0.0001
    exponential_avg_factor = 1
    data_format = "NCHW"
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

    # Input 5: Different batch size
    x = np.random.rand(32, 28, 28, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.zeros(3).astype(np.float32)
    variance = np.ones(3).astype(np.float32)
    epsilon = 0.0001
    exponential_avg_factor = 1
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

    # Input 6: Smaller image size
    x = np.random.rand(1, 14, 14, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.zeros(3).astype(np.float32)
    variance = np.ones(3).astype(np.float32)
    epsilon = 0.0001
    exponential_avg_factor = 1
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

    # Input 7: Half type
    x = np.random.rand(1, 28, 28, 3).astype(np.float16)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.zeros(3).astype(np.float32)
    variance = np.ones(3).astype(np.float32)
    epsilon = 0.0001
    exponential_avg_factor = 1
    data_format = "NHWC"
    is_training = True
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

    # Input 8:  exponential_avg_factor != 1 during training
    x = np.random.rand(1, 28, 28, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.zeros(3).astype(np.float32)
    variance = np.ones(3).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 0.5
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

    # Input 9: float32 type, inference
    x = np.random.rand(1, 28, 28, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)
    epsilon = 0.0001
    exponential_avg_factor = 1
    data_format = "NHWC"
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

    # Input 10: different epsilon
    x = np.random.rand(1, 28, 28, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.zeros(3).astype(np.float32)
    variance = np.ones(3).astype(np.float32)
    epsilon = 0.1
    exponential_avg_factor = 1
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
generated_inputs["tf.raw_ops.FusedBatchNormV2"] = tf_raw_ops_fused_batch_norm_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FusedBatchNormV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedBatchNormV2'.")

check_valid('tf.raw_ops.FusedBatchNormV2', generated_inputs['tf.raw_ops.FusedBatchNormV2'], lib="tf", suffix=0)
