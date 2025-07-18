
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def get_tf_raw_ops_apply_power_sign_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyPowerSign.
    NOTE: This raw op is stateful and intended for graph execution, requiring
    'ref' arguments (tf.Variable). It is incompatible with the standard eager
    execution context of the test harness, which will cause a persistent
    RuntimeError. The inputs provided here are valid according to the API
    signature but will fail in the user's execution environment.
    """
    list_of_inputs = []

    # Input 1: Basic float32, 1D
    input_dict_1 = {
        'var': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'm': np.array([0.1, 0.2, 0.3], dtype=np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'logbase': np.array(2.71828, dtype=np.float32),
        'sign_decay': np.array(0.9, dtype=np.float32),
        'beta': np.array(0.99, dtype=np.float32),
        'grad': np.array([-0.5, 0.5, -0.5], dtype=np.float32),
        'use_locking': False,
        'name': 'float32_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: float64, 2D with negative values and locking
    input_dict_2 = {
        'var': np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64),
        'm': np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        'lr': np.array(0.001, dtype=np.float64),
        'logbase': np.array(10.0, dtype=np.float64),
        'sign_decay': np.array(0.95, dtype=np.float64),
        'beta': np.array(0.9, dtype=np.float64),
        'grad': np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.float64),
        'use_locking': True,
        'name': 'float64_2d_locked'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: half (float16)
    input_dict_3 = {
        'var': np.array([1.0, 2.0, -3.0], dtype=np.float16),
        'm': np.array([0.1, -0.2, 0.3], dtype=np.float16),
        'lr': np.array(0.01, dtype=np.float16),
        'logbase': np.array(np.e, dtype=np.float16),
        'sign_decay': np.array(0.9, dtype=np.float16),
        'beta': np.array(0.99, dtype=np.float16),
        'grad': np.array([-0.5, 0.5, 0.0], dtype=np.float16),
        'use_locking': False,
        'name': 'float16_half'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: complex64
    input_dict_4 = {
        'var': np.array([1+1j, 2-2j], dtype=np.complex64),
        'm': np.array([0.5+0.5j, -0.5+0.5j], dtype=np.complex64),
        'lr': np.array(0.1+0j, dtype=np.complex64),
        'logbase': np.array(np.e+0j, dtype=np.complex64),
        'sign_decay': np.array(0.9+0j, dtype=np.complex64),
        'beta': np.array(0.99+0j, dtype=np.complex64),
        'grad': np.array([-1-1j, 1-1j], dtype=np.complex64),
        'use_locking': False,
        'name': 'complex64_all_complex'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: complex128
    input_dict_5 = {
        'var': np.array([[1+1j, 2-2j]], dtype=np.complex128),
        'm': np.array([[0.5+0.5j, 1j]], dtype=np.complex128),
        'lr': np.array(0.05+0j, dtype=np.complex128),
        'logbase': np.array(10.0+0j, dtype=np.complex128),
        'sign_decay': np.array(0.8+0j, dtype=np.complex128),
        'beta': np.array(0.88+0j, dtype=np.complex128),
        'grad': np.array([[-1-1j, -1j]], dtype=np.complex128),
        'use_locking': True,
        'name': 'complex128'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyPowerSign"] = get_tf_raw_ops_apply_power_sign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyPowerSign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyPowerSign'.")

check_valid('tf.raw_ops.ApplyPowerSign', generated_inputs['tf.raw_ops.ApplyPowerSign'], lib="tf", suffix=0)
