
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def get_tf_raw_ops_sparse_accumulator_apply_gradient_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.SparseAccumulatorApplyGradient operation.
    NOTE: This operation is not supported in eager execution. The inputs provided here are
    syntactically valid for graph construction, but will raise a RuntimeError if
    executed eagerly.
    """
    list_of_inputs = []

    # Input 1: Basic float32 gradient
    input_dict_1 = {
        'has_known_shape': True,
        'name': 'apply_float32',
        'handle': np.array(['handle1'], dtype=object),
        'local_step': np.array(100, dtype=np.int64),
        'gradient_indices': np.array([1, 5, 9], dtype=np.int64),
        'gradient_values': np.array([0.1, 0.2, 0.3], dtype=np.float32),
        'gradient_shape': np.array([10], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: int32 gradient
    input_dict_2 = {
        'has_known_shape': True,
        'name': 'apply_int32',
        'handle': np.array(['handle2'], dtype=object),
        'local_step': np.array(1, dtype=np.int64),
        'gradient_indices': np.array([0], dtype=np.int64),
        'gradient_values': np.array([-10,], dtype=np.int32),
        'gradient_shape': np.array([5], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: float64 gradient, no optional name
    input_dict_3 = {
        'has_known_shape': True,
        'name': None,
        'handle': np.array(['handle3'], dtype=object),
        'local_step': np.array(999999, dtype=np.int64),
        'gradient_indices': np.array([10, 20], dtype=np.int64),
        'gradient_values': np.array([1.23e45, -4.56e-10], dtype=np.float64),
        'gradient_shape': np.array([100], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: has_known_shape is False
    input_dict_4 = {
        'has_known_shape': False,
        'name': 'unknown_shape',
        'handle': np.array(['handle4'], dtype=object),
        'local_step': np.array(0, dtype=np.int64),
        'gradient_indices': np.array([0, 1], dtype=np.int64),
        'gradient_values': np.array([1, 2], dtype=np.int16),
        'gradient_shape': np.array([-1], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: complex64 gradient
    input_dict_5 = {
        'has_known_shape': True,
        'name': 'apply_complex64',
        'handle': np.array(['handle5'], dtype=object),
        'local_step': np.array(5, dtype=np.int64),
        'gradient_indices': np.array([3, 7], dtype=np.int64),
        'gradient_values': np.array([1+2j, 3-4j], dtype=np.complex64),
        'gradient_shape': np.array([10], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty gradient
    input_dict_6 = {
        'has_known_shape': True,
        'name': 'empty_grad',
        'handle': np.array(['handle6'], dtype=object),
        'local_step': np.array(20, dtype=np.int64),
        'gradient_indices': np.array([], dtype=np.int64),
        'gradient_values': np.array([], dtype=np.float32),
        'gradient_shape': np.array([100], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: uint8 gradient
    input_dict_7 = {
        'has_known_shape': True,
        'name': 'uint8_grad',
        'handle': np.array(['handle7'], dtype=object),
        'local_step': np.array(33, dtype=np.int64),
        'gradient_indices': np.array([10, 20, 30], dtype=np.int64),
        'gradient_values': np.array([1, 255, 128], dtype=np.uint8),
        'gradient_shape': np.array([50], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Gradient with multi-dimensional values (slices)
    input_dict_8 = {
        'has_known_shape': True,
        'name': 'slice_grad',
        'handle': np.array(['handle8'], dtype=object),
        'local_step': np.array(7, dtype=np.int64),
        'gradient_indices': np.array([1, 3], dtype=np.int64),
        'gradient_values': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'gradient_shape': np.array([5, 2], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: int64 values
    input_dict_9 = {
        'has_known_shape': True,
        'name': 'int64_values_grad',
        'handle': np.array(['handle9'], dtype=object),
        'local_step': np.array(2**32, dtype=np.int64),
        'gradient_indices': np.array([1000, 500000], dtype=np.int64),
        'gradient_values': np.array([1, -1], dtype=np.int64),
        'gradient_shape': np.array([1000000], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: half (float16) gradient
    input_dict_10 = {
        'has_known_shape': True,
        'name': 'half_grad',
        'handle': np.array(['handle10'], dtype=object),
        'local_step': np.array(12, dtype=np.int64),
        'gradient_indices': np.array([0, 1, 2], dtype=np.int64),
        'gradient_values': np.array([0.5, -0.5, 100.0], dtype=np.float16),
        'gradient_shape': np.array([3], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseAccumulatorApplyGradient"] = get_tf_raw_ops_sparse_accumulator_apply_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseAccumulatorApplyGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseAccumulatorApplyGradient'.")

check_valid('tf.raw_ops.SparseAccumulatorApplyGradient', generated_inputs['tf.raw_ops.SparseAccumulatorApplyGradient'], lib="tf", suffix=0)
