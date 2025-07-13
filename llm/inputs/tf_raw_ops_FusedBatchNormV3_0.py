
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fusedbatchnormv3_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case (NHWC, training)
    x = np.random.rand(1, 28, 28, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.array([]).astype(np.float32)
    variance = np.array([]).astype(np.float32)

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": 0.001,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": True,
        "name": "batch_norm_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NCHW, inference
    x = np.random.rand(1, 3, 28, 28).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": 0.00001,
        "exponential_avg_factor": 0.5,
        "data_format": "NCHW",
        "is_training": False,
        "name": "batch_norm_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16, training
    x = np.random.rand(1, 16, 16, 8).astype(np.float16)
    scale = np.random.rand(8).astype(np.float16)
    offset = np.random.rand(8).astype(np.float16)
    mean = np.array([]).astype(np.float16)
    variance = np.array([]).astype(np.float16)
    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": 0.0001,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": True,
        "name": "batch_norm_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, training
    x = np.random.rand(1, 16, 16, 8).astype(np.float16)
    scale = np.random.rand(8).astype(np.float16)
    offset = np.random.rand(8).astype(np.float16)
    mean = np.array([]).astype(np.float16)
    variance = np.array([]).astype(np.float16)
    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": 0.0001,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": True,
        "name": "batch_norm_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger batch size, NHWC
    x = np.random.rand(4, 32, 32, 16).astype(np.float32)
    scale = np.random.rand(16).astype(np.float32)
    offset = np.random.rand(16).astype(np.float32)
    mean = np.array([]).astype(np.float32)
    variance = np.array([]).astype(np.float32)

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": 0.0001,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": True,
        "name": "batch_norm_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Smaller image size, NCHW
    x = np.random.rand(1, 3, 8, 8).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": 0.0001,
        "exponential_avg_factor": 1.0,
        "data_format": "NCHW",
        "is_training": False,
        "name": "batch_norm_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different epsilon
    x = np.random.rand(1, 28, 28, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.array([]).astype(np.float32)
    variance = np.array([]).astype(np.float32)

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": 1e-8,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": True,
        "name": "batch_norm_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different exponential_avg_factor
    x = np.random.rand(1, 28, 28, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.array([]).astype(np.float32)
    variance = np.array([]).astype(np.float32)

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": 0.0001,
        "exponential_avg_factor": 0.9,
        "data_format": "NHWC",
        "is_training": True,
        "name": "batch_norm_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: NDHWC, Training
    x = np.random.rand(1, 16, 16, 16, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.array([]).astype(np.float32)
    variance = np.array([]).astype(np.float32)

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": 0.0001,
        "exponential_avg_factor": 1.0,
        "data_format": "NDHWC",
        "is_training": True,
        "name": "batch_norm_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: NCDHW, Inference
    x = np.random.rand(1, 3, 16, 16, 16).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)

    input_dict = {
        "x": x,
        "scale": scale,
        "offset": offset,
        "mean": mean,
        "variance": variance,
        "epsilon": 0.0001,
        "exponential_avg_factor": 1.0,
        "data_format": "NCDHW",
        "is_training": False,
        "name": "batch_norm_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FusedBatchNormV3"] = tf_raw_ops_fusedbatchnormv3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FusedBatchNormV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedBatchNormV3'.")

check_valid('tf.raw_ops.FusedBatchNormV3', generated_inputs['tf.raw_ops.FusedBatchNormV3'], lib="tf", suffix=0)
