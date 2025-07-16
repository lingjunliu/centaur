
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fused_batch_norm_v3_inputs():
    list_of_inputs = []

    # Input 1
    x = np.random.randn(1, 32, 32, 3).astype(np.float32)
    scale = np.random.randn(3).astype(np.float32)
    offset = np.random.randn(3).astype(np.float32)
    mean = np.array([]).astype(np.float32)
    variance = np.array([]).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 1.0
    data_format = "NHWC"
    is_training = True
    name = "batchnorm1"

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
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    scale = np.random.randn(3).astype(np.float32)
    offset = np.random.randn(3).astype(np.float32)
    mean = np.array([]).astype(np.float32)
    variance = np.array([]).astype(np.float32)
    epsilon = 0.00001
    exponential_avg_factor = 0.5
    data_format = "NCHW"
    is_training = True
    name = "batchnorm2"

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
    x = np.random.randn(2, 64, 64, 1).astype(np.float32)
    scale = np.random.randn(1).astype(np.float32)
    offset = np.random.randn(1).astype(np.float32)
    mean = np.array([]).astype(np.float32)
    variance = np.array([]).astype(np.float32)
    epsilon = 0.1
    exponential_avg_factor = 0.9
    data_format = "NHWC"
    is_training = True
    name = "batchnorm3"

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

    # Input 4 (inference)
    x = np.random.randn(1, 32, 32, 3).astype(np.float32)
    scale = np.random.randn(3).astype(np.float32)
    offset = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    variance = np.random.randn(3).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 1.0
    data_format = "NHWC"
    is_training = False
    name = "batchnorm4"

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


    # Input 5 (NDHWC)
    x = np.random.randn(1, 16, 16, 16, 3).astype(np.float32)
    scale = np.random.randn(3).astype(np.float32)
    offset = np.random.randn(3).astype(np.float32)
    mean = np.array([]).astype(np.float32)
    variance = np.array([]).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 1.0
    data_format = "NDHWC"
    is_training = True
    name = "batchnorm7"

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

    # Input 6 (NCDHW)
    x = np.random.randn(1, 3, 16, 16, 16).astype(np.float32)
    scale = np.random.randn(3).astype(np.float32)
    offset = np.random.randn(3).astype(np.float32)
    mean = np.array([]).astype(np.float32)
    variance = np.array([]).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 1.0
    data_format = "NCDHW"
    is_training = True
    name = "batchnorm8"

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

    # Input 7, different shape for scale and offset
    x = np.random.randn(1, 32, 32, 5).astype(np.float32)
    scale = np.random.randn(5).astype(np.float32)
    offset = np.random.randn(5).astype(np.float32)
    mean = np.array([]).astype(np.float32)
    variance = np.array([]).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 1.0
    data_format = "NHWC"
    is_training = True
    name = "batchnorm9"

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

     # Input 8, smaller epsilon
    x = np.random.randn(1, 32, 32, 3).astype(np.float32)
    scale = np.random.randn(3).astype(np.float32)
    offset = np.random.randn(3).astype(np.float32)
    mean = np.array([]).astype(np.float32)
    variance = np.array([]).astype(np.float32)
    epsilon = 1e-12
    exponential_avg_factor = 1.0
    data_format = "NHWC"
    is_training = True
    name = "batchnorm10"

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

    # Input 9 (float16)
    x = np.random.randn(1, 32, 32, 3).astype(np.float16)
    scale = np.random.randn(3).astype(np.float16)
    offset = np.random.randn(3).astype(np.float16)
    mean = np.array([]).astype(np.float16)
    variance = np.array([]).astype(np.float16)
    epsilon = 0.001
    exponential_avg_factor = 1.0
    data_format = "NHWC"
    is_training = True
    name = "batchnorm5"

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

    # Input 10 (NHWC, is_training=False)
    x = np.random.randn(1, 32, 32, 3).astype(np.float32)
    scale = np.random.randn(3).astype(np.float32)
    offset = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    variance = np.random.randn(3).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 1.0
    data_format = "NHWC"
    is_training = False
    name = "batchnorm11"

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
generated_inputs["tf.raw_ops.FusedBatchNormV3"] = tf_raw_ops_fused_batch_norm_v3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FusedBatchNormV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedBatchNormV3'.")

check_valid('tf.raw_ops.FusedBatchNormV3', generated_inputs['tf.raw_ops.FusedBatchNormV3'], lib="tf", suffix=0)
