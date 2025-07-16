
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fused_batch_norm_inputs():
    list_of_inputs = []

    # Input 1
    x = np.random.rand(1, 28, 28, 3).astype(np.float32)
    scale = np.random.rand(3).astype(np.float32)
    offset = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    variance = np.random.rand(3).astype(np.float32)
    epsilon = 0.001
    exponential_avg_factor = 0.5
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
    list_of_inputs.append(input_dict)

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
