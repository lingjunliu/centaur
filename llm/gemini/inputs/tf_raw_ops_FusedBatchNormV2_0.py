
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FusedBatchNormV2_inputs():
    list_of_inputs = []

    # Input 1: Training, NHWC, float32, basic
    input_dict = {
        "x": np.random.randn(2, 3, 3, 4).astype(np.float32),
        "scale": np.random.randn(4).astype(np.float32),
        "offset": np.random.randn(4).astype(np.float32),
        "mean": np.array([], dtype=np.float32),
        "variance": np.array([], dtype=np.float32),
        "epsilon": 0.0001,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": True,
        "name": "bn_train_nhwc_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Inference, NHWC, float32, basic
    input_dict = {
        "x": np.random.randn(2, 3, 3, 4).astype(np.float32),
        "scale": np.random.randn(4).astype(np.float32),
        "offset": np.random.randn(4).astype(np.float32),
        "mean": np.random.randn(4).astype(np.float32),
        "variance": np.abs(np.random.randn(4)).astype(np.float32),
        "epsilon": 0.001,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": False,
        "name": "bn_infer_nhwc_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Training, NCHW, float32
    input_dict = {
        "x": np.random.randn(2, 4, 3, 3).astype(np.float32),
        "scale": np.random.randn(4).astype(np.float32),
        "offset": np.random.randn(4).astype(np.float32),
        "mean": np.array([], dtype=np.float32),
        "variance": np.array([], dtype=np.float32),
        "epsilon": 0.0001,
        "exponential_avg_factor": 1.0,
        "data_format": "NCHW",
        "is_training": True,
        "name": "bn_train_nchw_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Inference, NCHW, float32
    input_dict = {
        "x": np.random.randn(2, 4, 3, 3).astype(np.float32),
        "scale": np.random.randn(4).astype(np.float32),
        "offset": np.random.randn(4).astype(np.float32),
        "mean": np.random.randn(4).astype(np.float32),
        "variance": np.abs(np.random.randn(4)).astype(np.float32),
        "epsilon": 1e-5,
        "exponential_avg_factor": 0.1,
        "data_format": "NCHW",
        "is_training": False,
        "name": "bn_infer_nchw_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Training, NHWC, float16
    input_dict = {
        "x": np.random.randn(1, 2, 2, 3).astype(np.float16),
        "scale": np.random.randn(3).astype(np.float32),
        "offset": np.random.randn(3).astype(np.float32),
        "mean": np.array([], dtype=np.float32),
        "variance": np.array([], dtype=np.float32),
        "epsilon": 0.0001,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": True,
        "name": "bn_train_nhwc_f16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Inference, NHWC, float16
    input_dict = {
        "x": np.random.randn(1, 2, 2, 3).astype(np.float16),
        "scale": np.random.randn(3).astype(np.float32),
        "offset": np.random.randn(3).astype(np.float32),
        "mean": np.random.randn(3).astype(np.float32),
        "variance": np.abs(np.random.randn(3)).astype(np.float32),
        "epsilon": 1e-4,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": False,
        "name": "bn_infer_nhwc_f16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Training, NHWC, larger batch
    input_dict = {
        "x": np.random.randn(8, 16, 16, 32).astype(np.float32),
        "scale": np.random.randn(32).astype(np.float32),
        "offset": np.random.randn(32).astype(np.float32),
        "mean": np.array([], dtype=np.float32),
        "variance": np.array([], dtype=np.float32),
        "epsilon": 1e-3,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": True,
        "name": "bn_train_large_nhwc"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Training, NCHW, larger batch, float16
    input_dict = {
        "x": np.random.randn(4, 16, 8, 8).astype(np.float16),
        "scale": np.random.randn(16).astype(np.float32),
        "offset": np.random.randn(16).astype(np.float32),
        "mean": np.array([], dtype=np.float32),
        "variance": np.array([], dtype=np.float32),
        "epsilon": 1e-4,
        "exponential_avg_factor": 1.0,
        "data_format": "NCHW",
        "is_training": True,
        "name": "bn_train_large_nchw_f16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Inference, NHWC, zero variance
    input_dict = {
        "x": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "scale": np.random.randn(2).astype(np.float32),
        "offset": np.random.randn(2).astype(np.float32),
        "mean": np.random.randn(2).astype(np.float32),
        "variance": np.zeros(2).astype(np.float32),
        "epsilon": 1e-5,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": False,
        "name": "bn_infer_zero_var"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Training, NCHW, minimal dimensions
    input_dict = {
        "x": np.random.randn(1, 1, 1, 1).astype(np.float32),
        "scale": np.random.randn(1).astype(np.float32),
        "offset": np.random.randn(1).astype(np.float32),
        "mean": np.array([], dtype=np.float32),
        "variance": np.array([], dtype=np.float32),
        "epsilon": 1e-4,
        "exponential_avg_factor": 1.0,
        "data_format": "NCHW",
        "is_training": True,
        "name": "bn_train_minimal"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FusedBatchNormV2"] = tf_raw_ops_FusedBatchNormV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FusedBatchNormV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedBatchNormV2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.FusedBatchNormV2', generated_inputs['tf.raw_ops.FusedBatchNormV2'], lib="tf", suffix=0)
