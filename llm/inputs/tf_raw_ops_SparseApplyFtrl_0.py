
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_SparseApplyFtrl_inputs():
    # This op modifies its inputs ('var', 'accum', 'linear') and is stateful.
    # In eager execution, this typically requires tf.Variable inputs.
    # However, the testing harness seems to fail on tf.Variable objects with
    # "AttributeError: 'ResourceVariable' object has no attribute 'size'".
    # To resolve this specific error, we provide numpy arrays, which the harness
    # can process. This may lead to a different error during the actual API call
    # ("RuntimeError: ... op does not support eager execution"), indicating a
    # fundamental incompatibility between the stateful op and the test setup.
    # We are addressing the error presented in the traceback.

    list_of_inputs = []

    # Input 1: Basic float32, 2D
    var_dtype_1 = np.float32
    var_1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=var_dtype_1)
    input_dict_1 = {
        'var': var_1,
        'accum': np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]], dtype=var_dtype_1),
        'linear': np.zeros_like(var_1),
        'grad': np.array([[0.5, 0.2], [-0.1, 0.3]], dtype=var_dtype_1),
        'indices': np.array([0, 2], dtype=np.int32),
        'lr': np.array(0.01, dtype=var_dtype_1),
        'l1': np.array(0.1, dtype=var_dtype_1),
        'l2': np.array(0.001, dtype=var_dtype_1),
        'lr_power': np.array(-0.5, dtype=var_dtype_1),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': "test_basic_float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: float64, 1D, with locking
    var_dtype_2 = np.float64
    var_2 = np.array([-1.0, -2.0, -3.0], dtype=var_dtype_2)
    input_dict_2 = {
        'var': var_2,
        'accum': np.array([0.2, 0.2, 0.2], dtype=var_dtype_2),
        'linear': np.array([0.1, -0.1, 0.2], dtype=var_dtype_2),
        'indices': np.array([1], dtype=np.int64),
        'grad': np.array([-0.5], dtype=var_dtype_2),
        'lr': np.array(0.1, dtype=var_dtype_2),
        'l1': np.array(1.0, dtype=var_dtype_2),
        'l2': np.array(0.5, dtype=var_dtype_2),
        'lr_power': np.array(-1.0, dtype=var_dtype_2),
        'use_locking': True,
        'multiply_linear_by_lr': True,
        'name': "test_float64_1d_locking"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: float16 (half)
    var_dtype_3 = np.float16
    var_3 = np.array([[1.0, 2.0]], dtype=var_dtype_3)
    input_dict_3 = {
        'var': var_3,
        'accum': np.full((1, 2), 0.1, dtype=var_dtype_3),
        'linear': np.zeros((1, 2), dtype=var_dtype_3),
        'indices': np.array([0], dtype=np.int32),
        'grad': np.array([[0.5, -0.5]], dtype=var_dtype_3),
        'lr': np.array(0.01, dtype=var_dtype_3),
        'l1': np.array(0.2, dtype=var_dtype_3),
        'l2': np.array(0.3, dtype=var_dtype_3),
        'lr_power': np.array(-0.6, dtype=var_dtype_3),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': "test_float16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4: Empty indices (should result in no update)
    var_dtype_4 = np.float32
    var_4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=var_dtype_4)
    input_dict_4 = {
        'var': var_4,
        'accum': np.full_like(var_4, 0.1),
        'linear': np.zeros_like(var_4),
        'indices': np.array([], dtype=np.int32),
        'grad': np.empty(shape=(0, 2), dtype=var_dtype_4),
        'lr': np.array(0.01, dtype=var_dtype_4),
        'l1': np.array(0.1, dtype=var_dtype_4),
        'l2': np.array(0.001, dtype=var_dtype_4),
        'lr_power': np.array(-0.5, dtype=var_dtype_4),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': "test_empty_indices"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Zero regularization
    var_dtype_5 = np.float32
    var_5 = np.array([[10.0]], dtype=var_dtype_5)
    input_dict_5 = {
        'var': var_5,
        'accum': np.array([[1.0]], dtype=var_dtype_5),
        'linear': np.array([[0.5]], dtype=var_dtype_5),
        'grad': np.array([[-2.0]], dtype=var_dtype_5),
        'indices': np.array([0], dtype=np.int32),
        'lr': np.array(0.1, dtype=var_dtype_5),
        'l1': np.array(0.0, dtype=var_dtype_5),
        'l2': np.array(0.0, dtype=var_dtype_5),
        'lr_power': np.array(-0.5, dtype=var_dtype_5),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': "test_zero_regularization"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyFtrl"] = tf_raw_ops_SparseApplyFtrl_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyFtrl' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyFtrl'.")

check_valid('tf.raw_ops.SparseApplyFtrl', generated_inputs['tf.raw_ops.SparseApplyFtrl'], lib="tf", suffix=0)
