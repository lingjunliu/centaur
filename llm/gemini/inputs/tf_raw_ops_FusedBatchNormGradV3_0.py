
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FusedBatchNormGradV3_inputs():
    list_of_inputs = []

    # Case 1: NHWC, float32, is_training=True
    input_dict = {
        'epsilon': 0.0001,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'fused_batch_norm_grad_1',
        'y_backprop': np.random.randn(2, 3, 4, 5).astype(np.float32),
        'x': np.random.randn(2, 3, 4, 5).astype(np.float32),
        'scale': np.random.randn(5).astype(np.float32),
        'reserve_space_1': np.random.randn(5).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(5).astype(np.float32)) + 0.1,
        'reserve_space_3': np.random.randn(5).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: NHWC, float16, is_training=True
    input_dict = {
        'epsilon': 0.0001,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'fused_batch_norm_grad_2',
        'y_backprop': np.random.randn(1, 2, 2, 3).astype(np.float16),
        'x': np.random.randn(1, 2, 2, 3).astype(np.float16),
        'scale': np.random.randn(3).astype(np.float32),
        'reserve_space_1': np.random.randn(3).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(3).astype(np.float32)) + 0.1,
        'reserve_space_3': np.random.randn(3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: NCHW, float32, is_training=True
    input_dict = {
        'epsilon': 1e-4,
        'data_format': 'NCHW',
        'is_training': True,
        'name': 'fused_batch_norm_grad_3',
        'y_backprop': np.random.randn(2, 3, 2, 2).astype(np.float32),
        'x': np.random.randn(2, 3, 2, 2).astype(np.float32),
        'scale': np.random.randn(3).astype(np.float32),
        'reserve_space_1': np.random.randn(3).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(3).astype(np.float32)) + 0.1,
        'reserve_space_3': np.random.randn(3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: NHWC, float32, is_training=False, reserve_space_3 is empty
    input_dict = {
        'epsilon': 1e-4,
        'data_format': 'NHWC',
        'is_training': False,
        'name': 'fused_batch_norm_grad_4',
        'y_backprop': np.random.randn(2, 3, 4, 4).astype(np.float32),
        'x': np.random.randn(2, 3, 4, 4).astype(np.float32),
        'scale': np.random.randn(4).astype(np.float32),
        'reserve_space_1': np.random.randn(4).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(4).astype(np.float32)) + 0.1,
        'reserve_space_3': np.array([], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: NHWC, float16, is_training=False, reserve_space_3 is empty
    input_dict = {
        'epsilon': 1e-5,
        'data_format': 'NHWC',
        'is_training': False,
        'name': 'fused_batch_norm_grad_5',
        'y_backprop': np.random.randn(1, 2, 2, 2).astype(np.float16),
        'x': np.random.randn(1, 2, 2, 2).astype(np.float16),
        'scale': np.random.randn(2).astype(np.float32),
        'reserve_space_1': np.random.randn(2).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(2).astype(np.float32)) + 0.1,
        'reserve_space_3': np.array([], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: NHWC, float32, small size, is_training=True
    input_dict = {
        'epsilon': 1e-5,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'fused_batch_norm_grad_6',
        'y_backprop': np.random.randn(1, 1, 1, 1).astype(np.float32),
        'x': np.random.randn(1, 1, 1, 1).astype(np.float32),
        'scale': np.random.randn(1).astype(np.float32),
        'reserve_space_1': np.random.randn(1).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(1).astype(np.float32)) + 0.1,
        'reserve_space_3': np.random.randn(1).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: NHWC, float32, negative inputs, is_training=True
    input_dict = {
        'epsilon': 1e-4,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'fused_batch_norm_grad_7',
        'y_backprop': -np.random.randn(2, 2, 2, 2).astype(np.float32),
        'x': -np.random.randn(2, 2, 2, 2).astype(np.float32),
        'scale': np.random.randn(2).astype(np.float32),
        'reserve_space_1': np.random.randn(2).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(2).astype(np.float32)) + 0.1,
        'reserve_space_3': np.random.randn(2).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: NCHW, float32, larger scale, is_training=True
    input_dict = {
        'epsilon': 0.001,
        'data_format': 'NCHW',
        'is_training': True,
        'name': 'fused_batch_norm_grad_8',
        'y_backprop': np.random.randn(4, 8, 4, 4).astype(np.float32),
        'x': np.random.randn(4, 8, 4, 4).astype(np.float32),
        'scale': np.random.randn(8).astype(np.float32),
        'reserve_space_1': np.random.randn(8).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(8).astype(np.float32)) + 0.1,
        'reserve_space_3': np.random.randn(8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: NHWC, float16, larger scale, is_training=True
    input_dict = {
        'epsilon': 0.0001,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'fused_batch_norm_grad_9',
        'y_backprop': np.random.randn(3, 3, 3, 3).astype(np.float16),
        'x': np.random.randn(3, 3, 3, 3).astype(np.float16),
        'scale': np.random.randn(3).astype(np.float32),
        'reserve_space_1': np.random.randn(3).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(3).astype(np.float32)) + 0.1,
        'reserve_space_3': np.random.randn(3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: NHWC, float32, is_training=False, reserve_space_3 is empty
    input_dict = {
        'epsilon': 0.0001,
        'data_format': 'NHWC',
        'is_training': False,
        'name': 'fused_batch_norm_grad_10',
        'y_backprop': np.random.randn(3, 5, 5, 4).astype(np.float32),
        'x': np.random.randn(3, 5, 5, 4).astype(np.float32),
        'scale': np.random.randn(4).astype(np.float32),
        'reserve_space_1': np.random.randn(4).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(4).astype(np.float32)) + 0.1,
        'reserve_space_3': np.array([], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FusedBatchNormGradV3"] = tf_raw_ops_FusedBatchNormGradV3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FusedBatchNormGradV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedBatchNormGradV3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.FusedBatchNormGradV3', generated_inputs['tf.raw_ops.FusedBatchNormGradV3'], lib="tf", suffix=0)
