
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_sparse_apply_ftrl_inputs():
    """
    This function generates a list of valid inputs for the
    tf.raw_ops.SparseApplyFtrl operation.
    The recurring "RuntimeError: sparse_apply_ftrl op does not support eager
    execution" is fundamental. The op requires mutable tf.Variable inputs
    (refs), which are part of TensorFlow's graph mode, while the execution
    environment uses immutable tf.Tensor objects from eager mode. This issue
    cannot be resolved by changing numpy input values. The following inputs
    are provided as a best-effort attempt to supply dimensionally and
    numerically correct data according to the API's contract.
    """
    list_of_inputs = []

    # Input 1: Basic 1D float32 case
    input_dict_1 = {
        'var': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'accum': np.array([0.1, 0.1, 0.1], dtype=np.float32),
        'linear': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        'grad': np.array([0.5], dtype=np.float32),
        'indices': np.array([1], dtype=np.int32),
        'lr': np.array(0.01, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.01, dtype=np.float32),
        'lr_power': np.array(-0.5, dtype=np.float32),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': 'basic_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D float32 case with multiple indices and locking
    input_dict_2 = {
        'var': np.ones((5, 3), dtype=np.float32),
        'accum': np.full((5, 3), 0.1, dtype=np.float32),
        'linear': np.zeros((5, 3), dtype=np.float32),
        'grad': np.random.randn(2, 3).astype(np.float32),
        'indices': np.array([0, 4], dtype=np.int32),
        'lr': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.0, dtype=np.float32),
        'l2': np.array(0.0, dtype=np.float32),
        'lr_power': np.array(-0.5, dtype=np.float32),
        'use_locking': True,
        'multiply_linear_by_lr': False,
        'name': 'basic_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 1D float64 case with multiply_linear_by_lr
    input_dict_3 = {
        'var': np.arange(5, dtype=np.float64),
        'accum': np.full((5,), 0.2, dtype=np.float64),
        'linear': np.zeros((5,), dtype=np.float64),
        'grad': np.array([-0.1, 0.2], dtype=np.float64),
        'indices': np.array([2, 3], dtype=np.int64),
        'lr': np.array(0.05, dtype=np.float64),
        'l1': np.array(0.0, dtype=np.float64),
        'l2': np.array(1.0, dtype=np.float64),
        'lr_power': np.array(-0.5, dtype=np.float64),
        'use_locking': False,
        'multiply_linear_by_lr': True,
        'name': 'basic_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Empty update (edge case with zero indices)
    input_dict_4 = {
        'var': np.ones((4, 2), dtype=np.float32),
        'accum': np.ones((4, 2), dtype=np.float32),
        'linear': np.zeros((4, 2), dtype=np.float32),
        'grad': np.empty((0, 2), dtype=np.float32),
        'indices': np.array([], dtype=np.int32),
        'lr': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.1, dtype=np.float32),
        'lr_power': np.array(-0.5, dtype=np.float32),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': 'empty_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: half precision (float16)
    input_dict_5 = {
        'var': np.array([1.0, 2.0, 3.0], dtype=np.half),
        'accum': np.array([0.1, 0.1, 0.1], dtype=np.half),
        'linear': np.array([0.5, 0.5, 0.5], dtype=np.half),
        'grad': np.array([0.2], dtype=np.half),
        'indices': np.array([1], dtype=np.int32),
        'lr': np.array(0.01, dtype=np.half),
        'l1': np.array(0.1, dtype=np.half),
        'l2': np.array(0.0, dtype=np.half),
        'lr_power': np.array(-0.5, dtype=np.half),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': 'basic_half'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyFtrl"] = tf_raw_ops_sparse_apply_ftrl_inputs()

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
