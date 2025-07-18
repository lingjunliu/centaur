
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_sparse_apply_rmsprop_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.SparseApplyRMSProp.
    The op is stateful and known to have issues with eager execution if not
    used with tf.Variable. The inputs provided are numpy arrays with shapes
    that conform to the API's documentation, as this is a common source of
    errors. grad.shape must be (len(indices),) + var.shape[1:].
    """
    list_of_inputs = []

    # Input 1: Basic float32, 1D var
    input_dict_1 = {
        'use_locking': False,
        'name': 'basic_1d_float32',
        'var': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        'ms': np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        'mom': np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'rho': np.array(0.9, dtype=np.float32),
        'momentum': np.array(0.8, dtype=np.float32),
        'epsilon': np.array(1e-7, dtype=np.float32),
        'grad': np.array([0.5, 0.6, 0.7], dtype=np.float32),
        'indices': np.array([0, 2, 4], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic float32, 2D var
    input_dict_2 = {
        'use_locking': True,
        'name': 'basic_2d_locking',
        'var': np.arange(12, dtype=np.float32).reshape(4, 3),
        'ms': np.random.rand(4, 3).astype(np.float32),
        'mom': np.random.rand(4, 3).astype(np.float32),
        'lr': np.array(0.001, dtype=np.float32),
        'rho': np.array(0.95, dtype=np.float32),
        'momentum': np.array(0.9, dtype=np.float32),
        'epsilon': np.array(1e-8, dtype=np.float32),
        'grad': np.random.rand(2, 3).astype(np.float32),
        'indices': np.array([1, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: float64, 3D var
    input_dict_3 = {
        'use_locking': False,
        'name': 'float64_3d',
        'var': np.random.rand(5, 2, 2).astype(np.float64),
        'ms': np.random.rand(5, 2, 2).astype(np.float64),
        'mom': np.random.rand(5, 2, 2).astype(np.float64),
        'lr': np.array(0.1, dtype=np.float64),
        'rho': np.array(0.99, dtype=np.float64),
        'momentum': np.array(0.0, dtype=np.float64),
        'epsilon': np.array(1e-10, dtype=np.float64),
        'grad': np.random.rand(2, 2, 2).astype(np.float64),
        'indices': np.array([0, 3], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: float16, 2D var, negative values
    input_dict_4 = {
        'use_locking': False,
        'name': 'float16_2d_negative',
        'var': np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]], dtype=np.float16),
        'ms': np.ones((3, 2), dtype=np.float16) * 0.5,
        'mom': np.random.randn(3, 2).astype(np.float16) * 0.1,
        'lr': np.array(0.02, dtype=np.float16),
        'rho': np.array(0.8, dtype=np.float16),
        'momentum': np.array(0.85, dtype=np.float16),
        'epsilon': np.array(1e-4, dtype=np.float16),
        'grad': np.array([[-0.9, 0.8], [0.7, -0.6]], dtype=np.float16),
        'indices': np.array([0, 2], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Duplicate indices
    input_dict_5 = {
        'use_locking': False,
        'name': 'duplicate_indices',
        'var': np.random.rand(5, 2).astype(np.float32),
        'ms': np.random.rand(5, 2).astype(np.float32),
        'mom': np.random.rand(5, 2).astype(np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'rho': np.array(0.9, dtype=np.float32),
        'momentum': np.array(0.8, dtype=np.float32),
        'epsilon': np.array(1e-7, dtype=np.float32),
        'grad': np.random.rand(4, 2).astype(np.float32),
        'indices': np.array([1, 3, 1, 4], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Zero gradient
    input_dict_6 = {
        'use_locking': False,
        'name': 'zero_grad_2d',
        'var': np.arange(8, dtype=np.float32).reshape(4, 2),
        'ms': np.ones((4, 2), dtype=np.float32),
        'mom': np.ones((4, 2), dtype=np.float32) * 0.1,
        'lr': np.array(0.1, dtype=np.float32),
        'rho': np.array(0.9, dtype=np.float32),
        'momentum': np.array(0.9, dtype=np.float32),
        'epsilon': np.array(1e-7, dtype=np.float32),
        'grad': np.zeros((2, 2), dtype=np.float32),
        'indices': np.array([0, 2], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty indices
    input_dict_7 = {
        'use_locking': False,
        'name': 'empty_indices_3d',
        'var': np.random.rand(3, 3, 3).astype(np.float32),
        'ms': np.random.rand(3, 3, 3).astype(np.float32),
        'mom': np.random.rand(3, 3, 3).astype(np.float32),
        'lr': np.array(0.001, dtype=np.float32),
        'rho': np.array(0.95, dtype=np.float32),
        'momentum': np.array(0.9, dtype=np.float32),
        'epsilon': np.array(1e-8, dtype=np.float32),
        'grad': np.empty(shape=(0, 3, 3), dtype=np.float32),
        'indices': np.array([], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: bfloat16
    bfloat16_dtype = tf.bfloat16.as_numpy_dtype
    input_dict_8 = {
        'use_locking': False,
        'name': 'bfloat16_3d',
        'var': np.random.rand(3, 2, 2).astype(bfloat16_dtype),
        'ms': np.random.rand(3, 2, 2).astype(bfloat16_dtype),
        'mom': np.random.rand(3, 2, 2).astype(bfloat16_dtype),
        'lr': np.array(0.02, dtype=bfloat16_dtype),
        'rho': np.array(0.8, dtype=bfloat16_dtype),
        'momentum': np.array(0.85, dtype=bfloat16_dtype),
        'epsilon': np.array(1e-4, dtype=bfloat16_dtype),
        'grad': np.random.rand(1, 2, 2).astype(bfloat16_dtype),
        'indices': np.array([2], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: All indices updated
    input_dict_9 = {
        'use_locking': False,
        'name': 'all_indices_updated_2d',
        'var': np.random.rand(3, 2).astype(np.float32),
        'ms': np.random.rand(3, 2).astype(np.float32),
        'mom': np.random.rand(3, 2).astype(np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'rho': np.array(0.9, dtype=np.float32),
        'momentum': np.array(0.8, dtype=np.float32),
        'epsilon': np.array(1e-7, dtype=np.float32),
        'grad': np.random.rand(3, 2).astype(np.float32),
        'indices': np.array([0, 1, 2], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: No name provided
    input_dict_10 = {
        'use_locking': False,
        'name': None,
        'var': np.array([1.0, 2.0], dtype=np.float32),
        'ms': np.array([0.1, 0.2], dtype=np.float32),
        'mom': np.array([0.01, 0.02], dtype=np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'rho': np.array(0.9, dtype=np.float32),
        'momentum': np.array(0.8, dtype=np.float32),
        'epsilon': np.array(1e-7, dtype=np.float32),
        'grad': np.array([0.5], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyRMSProp"] = tf_raw_ops_sparse_apply_rmsprop_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyRMSProp'.")

check_valid('tf.raw_ops.SparseApplyRMSProp', generated_inputs['tf.raw_ops.SparseApplyRMSProp'], lib="tf", suffix=0)
