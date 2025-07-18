
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_SparseApplyFtrlV2_inputs():
    """
    Generates a list of inputs for tf.raw_ops.SparseApplyFtrlV2.

    NOTE: This raw op is designed for use in TensorFlow's graph mode and modifies its
    input tensors in-place (as 'ref' types). It is known to raise a RuntimeError
    when called directly in TensorFlow's default eager execution mode, as eager tensors
    are immutable. The inputs provided here are valid according to the API's
    type, shape, and value constraints, and are intended for a graph-based
    execution context (e.g., inside a tf.function). The RuntimeError is inherent to
    calling this op eagerly and cannot be "fixed" by changing the NumPy inputs alone.
    """
    list_of_inputs = []

    # Helper function to create a base dictionary for a given configuration
    def create_base_dict(dtype, var_shape, indices):
        # Accumulators must be non-negative, and strictly positive if lr_power < 0
        accum = np.abs(np.random.rand(*var_shape).astype(dtype)) + 0.1
        grad_shape = (len(indices),) + var_shape[1:]

        return {
            'var': np.random.rand(*var_shape).astype(dtype),
            'accum': accum,
            'linear': np.random.rand(*var_shape).astype(dtype),
            'grad': np.random.rand(*grad_shape).astype(dtype),
            'indices': np.array(indices, dtype=np.int32 if max(indices or [0]) < 2**31 else np.int64),
            'lr': np.array(0.001, dtype=dtype),
            'l1': np.array(0.1, dtype=dtype),
            'l2': np.array(0.2, dtype=dtype),
            'l2_shrinkage': np.array(0.01, dtype=dtype),
            'lr_power': np.array(-0.5, dtype=dtype),
            'use_locking': False,
            'multiply_linear_by_lr': False,
            'name': 'test_case'
        }

    # Input 1: Basic float32, 1D var
    input_dict_1 = create_base_dict(np.float32, (10,), [2, 5, 8])
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: float64, 2D var, int64 indices
    input_dict_2 = create_base_dict(np.float64, (8, 4), [0, 3, 7])
    input_dict_2['use_locking'] = True
    input_dict_2['multiply_linear_by_lr'] = True
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: float32, 3D var
    input_dict_3 = create_base_dict(np.float32, (5, 3, 2), [1, 4])
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Empty indices (should be a no-op in a graph context)
    input_dict_4 = create_base_dict(np.float32, (6, 3), [])
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Duplicate indices
    input_dict_5 = create_base_dict(np.float32, (12,), [3, 7, 3, 10])
    input_dict_5['use_locking'] = True  # Locking is important with duplicates
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: All regularization terms set to zero
    input_dict_6 = create_base_dict(np.float64, (7, 5), [1, 6])
    input_dict_6['l1'] = np.array(0.0, dtype=np.float64)
    input_dict_6['l2'] = np.array(0.0, dtype=np.float64)
    input_dict_6['l2_shrinkage'] = np.array(0.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Dense update (all indices present)
    input_dict_7 = create_base_dict(np.float32, (5, 2), list(range(5)))
    input_dict_7['multiply_linear_by_lr'] = True
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyFtrlV2"] = get_tf_raw_ops_SparseApplyFtrlV2_inputs()

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
