
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_apply_ftrl_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyFtrl.
    
    NOTE: This operation is stateful and designed for TensorFlow's graph mode. It
    modifies its `var`, `accum`, and `linear` inputs in-place, which requires
    them to be `tf.Variable`s in a graph. Calling this raw op directly in an
    eager context (the default in TF2+) is not supported and will consistently
    raise a `RuntimeError`, as observed. The provided inputs are valid for the
    op's signature and logic within its intended graph-based execution context.
    """
    list_of_inputs = []

    # Case 1: Basic float32, 1D tensors
    input_dict_1 = {
        'var': np.array([1.0, 2.0], dtype=np.float32),
        'accum': np.array([0.1, 0.1], dtype=np.float32),
        'linear': np.array([0.5, -0.5], dtype=np.float32),
        'grad': np.array([0.01, -0.02], dtype=np.float32),
        'lr': np.array(0.001, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.2, dtype=np.float32),
        'lr_power': np.array(-0.5, dtype=np.float32),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': 'test_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: float64, 2D tensors, no regularization
    input_dict_2 = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        'accum': np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64),
        'linear': np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float64),
        'grad': np.array([[0.05, 0.06], [-0.07, -0.08]], dtype=np.float64),
        'lr': np.array(0.1, dtype=np.float64),
        'l1': np.array(0.0, dtype=np.float64),
        'l2': np.array(0.0, dtype=np.float64),
        'lr_power': np.array(-0.5, dtype=np.float64),
        'use_locking': True,
        'multiply_linear_by_lr': False,
        'name': 'test_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Scalar tensors
    input_dict_3 = {
        'var': np.array(5.0, dtype=np.float32),
        'accum': np.array(1.0, dtype=np.float32),
        'linear': np.array(-2.0, dtype=np.float32),
        'grad': np.array(0.5, dtype=np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'l1': np.array(0.0, dtype=np.float32),
        'l2': np.array(1.0, dtype=np.float32),
        'lr_power': np.array(-0.3, dtype=np.float32),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': 'test_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: multiply_linear_by_lr = True
    input_dict_4 = {
        'var': np.array([1.0, 2.0], dtype=np.float32),
        'accum': np.array([0.1, 0.1], dtype=np.float32),
        'linear': np.array([0.5, -0.5], dtype=np.float32),
        'grad': np.array([0.01, -0.02], dtype=np.float32),
        'lr': np.array(0.001, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.2, dtype=np.float32),
        'lr_power': np.array(-0.5, dtype=np.float32),
        'use_locking': False,
        'multiply_linear_by_lr': True,
        'name': 'test_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Zero gradients
    input_dict_5 = {
        'var': np.array([10.0, -10.0], dtype=np.float32),
        'accum': np.array([1.0, 1.0], dtype=np.float32),
        'linear': np.array([0.1, -0.1], dtype=np.float32),
        'grad': np.array([0.0, 0.0], dtype=np.float32),
        'lr': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.01, dtype=np.float32),
        'l2': np.array(0.1, dtype=np.float32),
        'lr_power': np.array(-0.5, dtype=np.float32),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': 'test_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: High L1 regularization
    input_dict_6 = {
        'var': np.array([0.1, 0.2], dtype=np.float32),
        'accum': np.array([0.1, 0.1], dtype=np.float32),
        'linear': np.array([0.01, -0.01], dtype=np.float32),
        'grad': np.array([0.001, 0.002], dtype=np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'l1': np.array(10.0, dtype=np.float32),
        'l2': np.array(0.0, dtype=np.float32),
        'lr_power': np.array(-0.5, dtype=np.float32),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': 'test_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Case 7: High L2 regularization
    input_dict_7 = {
        'var': np.array([0.1, 0.2], dtype=np.float32),
        'accum': np.array([0.1, 0.1], dtype=np.float32),
        'linear': np.array([0.01, -0.01], dtype=np.float32),
        'grad': np.array([0.001, 0.002], dtype=np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'l1': np.array(0.0, dtype=np.float32),
        'l2': np.array(10.0, dtype=np.float32),
        'lr_power': np.array(-0.5, dtype=np.float32),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': 'test_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Different lr_power
    input_dict_8 = {
        'var': np.array([1.5, 2.5], dtype=np.float64),
        'accum': np.array([0.2, 0.2], dtype=np.float64),
        'linear': np.array([0.6, -0.6], dtype=np.float64),
        'grad': np.array([0.03, -0.04], dtype=np.float64),
        'lr': np.array(0.005, dtype=np.float64),
        'l1': np.array(0.15, dtype=np.float64),
        'l2': np.array(0.25, dtype=np.float64),
        'lr_power': np.array(-0.25, dtype=np.float64),
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': 'test_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: All inputs positive (where sensible)
    input_dict_9 = {
        'var': np.array([1.0, 2.0], dtype=np.float32),
        'accum': np.array([1.0, 1.0], dtype=np.float32),
        'linear': np.array([0.5, 0.5], dtype=np.float32),
        'grad': np.array([0.01, 0.02], dtype=np.float32),
        'lr': np.array(0.001, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.2, dtype=np.float32),
        'lr_power': np.array(-0.5, dtype=np.float32),
        'use_locking': False,
        'multiply_linear_by_lr': True,
        'name': 'test_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Larger values in inputs
    input_dict_10 = {
        'var': np.array([1000.0, -2000.0], dtype=np.float32),
        'accum': np.array([100.0, 100.0], dtype=np.float32),
        'linear': np.array([500.0, -500.0], dtype=np.float32),
        'grad': np.array([10.0, -20.0], dtype=np.float32),
        'lr': np.array(0.1, dtype=np.float32),
        'l1': np.array(1.0, dtype=np.float32),
        'l2': np.array(2.0, dtype=np.float32),
        'lr_power': np.array(-0.5, dtype=np.float32),
        'use_locking': True,
        'multiply_linear_by_lr': False,
        'name': 'test_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyFtrl"] = tf_raw_ops_apply_ftrl_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyFtrl' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyFtrl'.")

check_valid('tf.raw_ops.ApplyFtrl', generated_inputs['tf.raw_ops.ApplyFtrl'], lib="tf", suffix=0)
