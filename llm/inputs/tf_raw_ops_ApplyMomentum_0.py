
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_apply_momentum_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyMomentum.
    """
    list_of_inputs = []

    # Input 1: Basic float32, 1D
    input_dict_1 = {
        'var': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'accum': np.array([0.1, 0.2, 0.3], dtype=np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'grad': np.array([0.5, -0.5, 0.0], dtype=np.float32),
        'momentum': np.array(0.9, dtype=np.float32),
        'use_locking': False,
        'use_nesterov': False,
        'name': "apply_momentum_float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: float64, 2D, Nesterov enabled
    input_dict_2 = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        'accum': np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        'lr': np.array(0.001, dtype=np.float64),
        'grad': np.array([[-0.2, 0.3], [0.1, -0.4]], dtype=np.float64),
        'momentum': np.array(0.8, dtype=np.float64),
        'use_locking': False,
        'use_nesterov': True,
        'name': "apply_momentum_float64_nesterov"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: float16 (half), negative values
    input_dict_3 = {
        'var': np.array([-1.5, -2.5, -3.5], dtype=np.float16),
        'accum': np.zeros(3, dtype=np.float16),
        'lr': np.array(0.1, dtype=np.float16),
        'grad': np.array([-0.1, 0.2, -0.3], dtype=np.float16),
        'momentum': np.array(0.99, dtype=np.float16),
        'use_locking': False,
        'use_nesterov': False,
        'name': "apply_momentum_float16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: complex64, Nesterov and locking enabled
    input_dict_4 = {
        'var': np.array([1+2j, 3+4j], dtype=np.complex64),
        'accum': np.array([0.1+0.1j, 0.2+0.2j], dtype=np.complex64),
        'lr': np.array(0.01, dtype=np.complex64),
        'grad': np.array([0.5+0.2j, -0.3-0.1j], dtype=np.complex64),
        'momentum': np.array(0.9, dtype=np.complex64),
        'use_locking': True,
        'use_nesterov': True,
        'name': "apply_momentum_complex64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: complex128
    input_dict_5 = {
        'var': np.array([[1.0+2.0j]], dtype=np.complex128),
        'accum': np.zeros((1, 1), dtype=np.complex128),
        'lr': np.array(0.5, dtype=np.complex128),
        'grad': np.array([[0.1-0.1j]], dtype=np.complex128),
        'momentum': np.array(0.85, dtype=np.complex128),
        'use_locking': False,
        'use_nesterov': False,
        'name': "apply_momentum_complex128"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: High-dimensional tensors (float32)
    input_dict_6 = {
        'var': np.ones((1, 2, 3, 1), dtype=np.float32),
        'accum': np.zeros((1, 2, 3, 1), dtype=np.float32),
        'lr': np.array(0.001, dtype=np.float32),
        'grad': np.random.randn(1, 2, 3, 1).astype(np.float32),
        'momentum': np.array(0.95, dtype=np.float32),
        'use_locking': False,
        'use_nesterov': True,
        'name': "apply_momentum_high_dim"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Zero-valued tensors
    input_dict_7 = {
        'var': np.zeros((5, 5), dtype=np.float32),
        'accum': np.zeros((5, 5), dtype=np.float32),
        'lr': np.array(0.1, dtype=np.float32),
        'grad': np.zeros((5, 5), dtype=np.float32),
        'momentum': np.array(0.9, dtype=np.float32),
        'use_locking': False,
        'use_nesterov': False,
        'name': "apply_momentum_zeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: uint16
    input_dict_8 = {
        'var': np.array([100, 200], dtype=np.uint16),
        'accum': np.array([10, 20], dtype=np.uint16),
        'lr': np.array(1, dtype=np.uint16),
        'grad': np.array([5, 8], dtype=np.uint16),
        'momentum': np.array(1, dtype=np.uint16),
        'use_locking': True,
        'use_nesterov': False,
        'name': "apply_momentum_uint16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: int8
    input_dict_9 = {
        'var': np.array([-10, 20, -30], dtype=np.int8),
        'accum': np.array([1, 2, 3], dtype=np.int8),
        'lr': np.array(2, dtype=np.int8),
        'grad': np.array([-2, 1, 3], dtype=np.int8),
        'momentum': np.array(1, dtype=np.int8),
        'use_locking': False,
        'use_nesterov': True,
        'name': "apply_momentum_int8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: float32 with locking
    input_dict_10 = {
        'var': np.random.rand(4, 1).astype(np.float32),
        'accum': np.zeros((4, 1), dtype=np.float32),
        'lr': np.array(0.05, dtype=np.float32),
        'grad': np.random.rand(4, 1).astype(np.float32),
        'momentum': np.array(0.88, dtype=np.float32),
        'use_locking': True,
        'use_nesterov': False,
        'name': "apply_momentum_float32_locking"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyMomentum"] = tf_raw_ops_apply_momentum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyMomentum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyMomentum'.")

check_valid('tf.raw_ops.ApplyMomentum', generated_inputs['tf.raw_ops.ApplyMomentum'], lib="tf", suffix=0)
