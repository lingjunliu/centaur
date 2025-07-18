
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def get_tf_raw_ops_sparse_apply_adagrad_da_inputs():
    list_of_inputs = []

    # Case 1: float32, basic, 2D.
    input_dict_1 = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        'gradient_accumulator': np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]], dtype=np.float32),
        'gradient_squared_accumulator': np.array([[0.01, 0.01], [0.01, 0.01], [0.01, 0.01]], dtype=np.float32),
        'grad': np.array([[0.1, 0.2], [0.5, 0.6]], dtype=np.float32),
        'indices': np.array([0, 2], dtype=np.int32),
        'lr': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.01, dtype=np.float32),
        'l2': np.array(0.02, dtype=np.float32),
        'global_step': np.array(10, dtype=np.int64),
        'use_locking': False,
        'name': 'float32_basic'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: float64, 1D, use_locking=True.
    input_dict_2 = {
        'var': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        'gradient_accumulator': np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        'gradient_squared_accumulator': np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        'grad': np.array([-0.2, 0.3], dtype=np.float64),
        'indices': np.array([1, 3], dtype=np.int64),
        'lr': np.array(0.01, dtype=np.float64),
        'l1': np.array(0.0, dtype=np.float64),
        'l2': np.array(0.005, dtype=np.float64),
        'global_step': np.array(100, dtype=np.int64),
        'use_locking': True,
        'name': 'float64_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: float32, 3D.
    var_3d = np.arange(2 * 3 * 2).reshape(2, 3, 2).astype(np.float32)
    input_dict_3 = {
        'var': var_3d,
        'gradient_accumulator': np.full_like(var_3d, 0.1, dtype=np.float32),
        'gradient_squared_accumulator': np.full_like(var_3d, 0.01, dtype=np.float32),
        'grad': np.array([[[1.0, 1.1], [1.2, 1.3], [1.4, 1.5]]], dtype=np.float32),
        'indices': np.array([1], dtype=np.int32),
        'lr': np.array(1e-4, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.2, dtype=np.float32),
        'global_step': np.array(0, dtype=np.int64),
        'use_locking': False,
        'name': 'float32_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: half (float16).
    var_f16 = np.array([[1.0], [2.0], [3.0]], dtype=np.half)
    input_dict_4 = {
        'var': var_f16,
        'gradient_accumulator': np.zeros_like(var_f16, dtype=np.half),
        'gradient_squared_accumulator': np.ones_like(var_f16, dtype=np.half),
        'grad': np.array([[0.1], [-0.2], [0.3]], dtype=np.half),
        'indices': np.array([0, 1, 2], dtype=np.int64),
        'lr': np.array(0.5, dtype=np.half),
        'l1': np.array(0.0, dtype=np.half),
        'l2': np.array(0.0, dtype=np.half),
        'global_step': np.array(1, dtype=np.int64),
        'use_locking': True,
        'name': 'float16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: bfloat16.
    bf16_dtype = tf.bfloat16.as_numpy_dtype
    var_bf16 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=bf16_dtype)
    input_dict_5 = {
        'var': var_bf16,
        'gradient_accumulator': np.zeros_like(var_bf16),
        'gradient_squared_accumulator': np.ones_like(var_bf16),
        'grad': np.array([[-0.5, -0.6]], dtype=bf16_dtype),
        'indices': np.array([1], dtype=np.int32),
        'lr': np.array(0.1, dtype=bf16_dtype),
        'l1': np.array(0.2, dtype=bf16_dtype),
        'l2': np.array(0.3, dtype=bf16_dtype),
        'global_step': np.array(5, dtype=np.int64),
        'use_locking': False,
        'name': 'bfloat16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: empty indices (no-op).
    var_empty = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict_6 = {
        'var': var_empty,
        'gradient_accumulator': np.zeros_like(var_empty),
        'gradient_squared_accumulator': np.ones_like(var_empty),
        'grad': np.empty(shape=(0, 2), dtype=np.float32),
        'indices': np.array([], dtype=np.int32),
        'lr': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.1, dtype=np.float32),
        'global_step': np.array(10, dtype=np.int64),
        'use_locking': False,
        'name': 'empty_indices'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: complex64.
    var_c64 = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    input_dict_7 = {
        'var': var_c64,
        'gradient_accumulator': np.zeros_like(var_c64, dtype=np.complex64),
        'gradient_squared_accumulator': np.ones_like(var_c64, dtype=np.complex64),
        'grad': np.array([[0.1+0.1j, -0.2-0.2j]], dtype=np.complex64),
        'indices': np.array([1], dtype=np.int64),
        'lr': np.array(0.1, dtype=np.complex64),
        'l1': np.array(0.01, dtype=np.complex64),
        'l2': np.array(0.02, dtype=np.complex64),
        'global_step': np.array(2, dtype=np.int64),
        'use_locking': False,
        'name': 'complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: zero regularization.
    var_zero_reg = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict_8 = {
        'var': var_zero_reg,
        'gradient_accumulator': np.ones_like(var_zero_reg),
        'gradient_squared_accumulator': np.ones_like(var_zero_reg),
        'grad': np.array([[100.0, 100.0]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'lr': np.array(0.01, dtype=np.float32),
        'l1': np.array(0.0, dtype=np.float32),
        'l2': np.array(0.0, dtype=np.float32),
        'global_step': np.array(50, dtype=np.int64),
        'use_locking': True,
        'name': 'zero_reg'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyAdagradDA"] = get_tf_raw_ops_sparse_apply_adagrad_da_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyAdagradDA' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyAdagradDA'.")

check_valid('tf.raw_ops.SparseApplyAdagradDA', generated_inputs['tf.raw_ops.SparseApplyAdagradDA'], lib="tf", suffix=0)
