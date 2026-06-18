
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_FusedBatchNormV3_inputs():
    list_of_inputs = []

    # Input 1: NHWC, float32, is_training=True
    x = np.random.randn(2, 3, 3, 4).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)
    offset = np.random.randn(4).astype(np.float32)
    mean = np.array([], dtype=np.float32)
    variance = np.array([], dtype=np.float32)
    
    input_dict = {
        'epsilon': 0.0001,
        'exponential_avg_factor': 1.0,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'bn_case_1',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NCHW, float32, is_training=True
    x = np.random.randn(2, 4, 3, 3).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)
    offset = np.random.randn(4).astype(np.float32)
    mean = np.array([], dtype=np.float32)
    variance = np.array([], dtype=np.float32)
    
    input_dict = {
        'epsilon': 1e-5,
        'exponential_avg_factor': 1.0,
        'data_format': 'NCHW',
        'is_training': True,
        'name': 'bn_case_2',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NHWC, float32, is_training=False
    x = np.random.randn(2, 3, 3, 4).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)
    offset = np.random.randn(4).astype(np.float32)
    mean = np.random.randn(4).astype(np.float32)
    variance = np.abs(np.random.randn(4).astype(np.float32))
    
    input_dict = {
        'epsilon': 0.0001,
        'exponential_avg_factor': 1.0,
        'data_format': 'NHWC',
        'is_training': False,
        'name': 'bn_case_3',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NCHW, float32, is_training=False
    x = np.random.randn(2, 8, 4, 4).astype(np.float32)
    scale = np.random.randn(8).astype(np.float32)
    offset = np.random.randn(8).astype(np.float32)
    mean = np.random.randn(8).astype(np.float32)
    variance = np.abs(np.random.randn(8).astype(np.float32))
    
    input_dict = {
        'epsilon': 1e-4,
        'exponential_avg_factor': 1.0,
        'data_format': 'NCHW',
        'is_training': False,
        'name': 'bn_case_4',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NHWC, float16 (half) for x, is_training=True
    x = np.random.randn(4, 8, 8, 16).astype(np.float16)
    scale = np.random.randn(16).astype(np.float32)
    offset = np.random.randn(16).astype(np.float32)
    mean = np.array([], dtype=np.float32)
    variance = np.array([], dtype=np.float32)
    
    input_dict = {
        'epsilon': 0.0001,
        'exponential_avg_factor': 1.0,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'bn_case_5',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: NCHW, float16 (half) for x, is_training=True
    x = np.random.randn(4, 16, 8, 8).astype(np.float16)
    scale = np.random.randn(16).astype(np.float32)
    offset = np.random.randn(16).astype(np.float32)
    mean = np.array([], dtype=np.float32)
    variance = np.array([], dtype=np.float32)
    
    input_dict = {
        'epsilon': 0.0001,
        'exponential_avg_factor': 1.0,
        'data_format': 'NCHW',
        'is_training': True,
        'name': 'bn_case_6',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NHWC, float16 (half) for x, is_training=False
    x = np.random.randn(4, 8, 8, 16).astype(np.float16)
    scale = np.random.randn(16).astype(np.float32)
    offset = np.random.randn(16).astype(np.float32)
    mean = np.random.randn(16).astype(np.float32)
    variance = np.abs(np.random.randn(16).astype(np.float32))
    
    input_dict = {
        'epsilon': 0.0001,
        'exponential_avg_factor': 1.0,
        'data_format': 'NHWC',
        'is_training': False,
        'name': 'bn_case_7',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: NCHW, float16 (half) for x, is_training=False
    x = np.random.randn(4, 16, 8, 8).astype(np.float16)
    scale = np.random.randn(16).astype(np.float32)
    offset = np.random.randn(16).astype(np.float32)
    mean = np.random.randn(16).astype(np.float32)
    variance = np.abs(np.random.randn(16).astype(np.float32))
    
    input_dict = {
        'epsilon': 0.0001,
        'exponential_avg_factor': 1.0,
        'data_format': 'NCHW',
        'is_training': False,
        'name': 'bn_case_8',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: NHWC, large batch, float32, is_training=True
    x = np.random.randn(32, 16, 16, 32).astype(np.float32)
    scale = np.random.randn(32).astype(np.float32)
    offset = np.random.randn(32).astype(np.float32)
    mean = np.array([], dtype=np.float32)
    variance = np.array([], dtype=np.float32)
    
    input_dict = {
        'epsilon': 1e-5,
        'exponential_avg_factor': 1.0,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'bn_case_9',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: NHWC, large batch, float32, is_training=False
    x = np.random.randn(32, 16, 16, 32).astype(np.float32)
    scale = np.random.randn(32).astype(np.float32)
    offset = np.random.randn(32).astype(np.float32)
    mean = np.random.randn(32).astype(np.float32)
    variance = np.abs(np.random.randn(32).astype(np.float32))
    
    input_dict = {
        'epsilon': 1e-5,
        'exponential_avg_factor': 1.0,
        'data_format': 'NHWC',
        'is_training': False,
        'name': 'bn_case_10',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FusedBatchNormV3"] = tf_raw_ops_FusedBatchNormV3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FusedBatchNormV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedBatchNormV3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.FusedBatchNormV3', generated_inputs['tf.raw_ops.FusedBatchNormV3'], lib="tf", suffix=0)
