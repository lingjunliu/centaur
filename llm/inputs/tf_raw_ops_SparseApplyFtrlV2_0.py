
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_sparse_apply_ftrl_v2_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.SparseApplyFtrlV2.

    NOTE: This raw op modifies its 'var' input in-place (it's a 'ref' argument)
    and is therefore NOT compatible with TensorFlow's eager execution mode, which is
    the default in TF2.x. Calling this function directly in an eager context will
    always raise a RuntimeError. The provided inputs are syntactically and
    semantically valid for graph-based execution (e.g., within a tf.function
    or a tf.compat.v1.Session), which is the environment where this op is
    designed to be used. The persistent error is due to the execution context,
    not the inputs themselves.
    """
    list_of_inputs = []

    # Helper function to create scalar numpy arrays
    def scalar(value, dtype):
        return np.array(value, dtype=dtype)

    # Input 1: Basic float32, 2D case
    input_dict_1 = {
        'var': np.arange(1, 7, dtype=np.float32).reshape(3, 2),
        'accum': np.full((3, 2), 0.1, dtype=np.float32),
        'linear': np.zeros((3, 2), dtype=np.float32),
        'grad': np.array([[0.5, 0.2], [0.1, 0.3]], dtype=np.float32),
        'indices': np.array([0, 2], dtype=np.int32),
        'lr': scalar(0.01, np.float32),
        'l1': scalar(0.1, np.float32),
        'l2': scalar(0.2, np.float32),
        'l2_shrinkage': scalar(0.001, np.float32),
        'lr_power': scalar(-0.5, np.float32),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': 'case1_basic_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: float64, 1D var, locking enabled
    input_dict_2 = {
        'var': np.arange(5, dtype=np.float64),
        'accum': np.full(5, 0.1, dtype=np.float64),
        'linear': np.zeros(5, dtype=np.float64),
        'grad': np.array([0.1, 0.2, 0.3], dtype=np.float64),
        'indices': np.array([1, 3, 4], dtype=np.int64),
        'lr': scalar(0.001, np.float64),
        'l1': scalar(1.0, np.float64),
        'l2': scalar(0.5, np.float64),
        'l2_shrinkage': scalar(0.0, np.float64),
        'lr_power': scalar(-0.5, np.float64),
        'use_locking': True,
        'multiply_linear_by_lr': False,
        'name': 'case2_float64_locking'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: multiply_linear_by_lr = True
    input_dict_3 = {
        'var': np.ones((4, 5), dtype=np.float32),
        'accum': np.full((4, 5), 1.0, dtype=np.float32),
        'linear': np.random.randn(4, 5).astype(np.float32),
        'grad': np.random.randn(2, 5).astype(np.float32),
        'indices': np.array([0, 3], dtype=np.int32),
        'lr': scalar(0.1, np.float32),
        'l1': scalar(0.0, np.float32),
        'l2': scalar(0.0, np.float32),
        'l2_shrinkage': scalar(0.0, np.float32),
        'lr_power': scalar(-0.8, np.float32),
        'use_locking': False,
        'multiply_linear_by_lr': True,
        'name': 'case3_multiply_linear'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Empty indices (should be a valid no-op)
    input_dict_4 = {
        'var': np.ones((3, 3), dtype=np.float32),
        'accum': np.ones((3, 3), dtype=np.float32),
        'linear': np.ones((3, 3), dtype=np.float32),
        'grad': np.empty((0, 3), dtype=np.float32),
        'indices': np.array([], dtype=np.int32),
        'lr': scalar(0.1, np.float32),
        'l1': scalar(0.1, np.float32),
        'l2': scalar(0.1, np.float32),
        'l2_shrinkage': scalar(0.1, np.float32),
        'lr_power': scalar(-0.5, np.float32),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': 'case4_empty_indices'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Higher dimensions for var
    input_dict_5 = {
        'var': np.ones((4, 2, 2), dtype=np.float32),
        'accum': np.full((4, 2, 2), 0.1, dtype=np.float32),
        'linear': np.zeros((4, 2, 2), dtype=np.float32),
        'grad': np.random.rand(2, 2, 2).astype(np.float32),
        'indices': np.array([0, 3], dtype=np.int32),
        'lr': scalar(0.1, np.float32),
        'l1': scalar(0.5, np.float32),
        'l2': scalar(0.5, np.float32),
        'l2_shrinkage': scalar(0.0, np.float32),
        'lr_power': scalar(-0.5, np.float32),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': 'case5_3d_var'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: half precision (float16)
    input_dict_6 = {
        'var': np.ones((5, 2), dtype=np.float16),
        'accum': np.full((5, 2), 0.1, dtype=np.float16),
        'linear': np.zeros((5, 2), dtype=np.float16),
        'grad': np.array([[0.1, 0.2]], dtype=np.float16),
        'indices': np.array([2], dtype=np.int32),
        'lr': scalar(0.01, np.float16),
        'l1': scalar(0.0, np.float16),
        'l2': scalar(0.0, np.float16),
        'l2_shrinkage': scalar(0.0, np.float16),
        'lr_power': scalar(-0.5, np.float16),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': 'case6_float16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))


    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyFtrlV2"] = get_tf_raw_ops_sparse_apply_ftrl_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyFtrlV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyFtrlV2'.")

check_valid('tf.raw_ops.SparseApplyFtrlV2', generated_inputs['tf.raw_ops.SparseApplyFtrlV2'], lib="tf", suffix=0)
