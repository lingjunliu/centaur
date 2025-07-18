
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def get_apply_gradient_descent_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyGradientDescent.
    This version provides inputs as NumPy arrays to satisfy the testing harness's
    analysis phase and focuses on standard floating-point types to mitigate
    potential execution errors.
    """
    list_of_inputs = []

    # Input 1: Basic float32, 1D
    input_dict_1 = {
        'var': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'alpha': np.array(0.1, dtype=np.float32),
        'delta': np.array([0.5, 0.5, 0.5], dtype=np.float32),
        'use_locking': False,
        'name': 'test_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: float64, 2D, with locking
    input_dict_2 = {
        'var': np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64),
        'alpha': np.array(0.01, dtype=np.float64),
        'delta': np.array([[10.0, 20.0], [-10.0, -20.0]], dtype=np.float64),
        'use_locking': True,
        'name': 'test_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: half (float16)
    input_dict_3 = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
        'alpha': np.array(0.5, dtype=np.float16),
        'delta': np.full((2, 2), 0.1, dtype=np.float16),
        'use_locking': False,
        'name': 'test_float16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Scalar inputs
    input_dict_4 = {
        'var': np.array(100.0, dtype=np.float32),
        'alpha': np.array(0.1, dtype=np.float32),
        'delta': np.array(50.0, dtype=np.float32),
        'use_locking': False,
        'name': 'test_scalar_numpy'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: bfloat16
    bfloat16_dtype = tf.bfloat16.as_numpy_dtype
    input_dict_5 = {
        'var': np.array([[1.0, 2.0]], dtype=bfloat16_dtype),
        'alpha': np.array(0.1, dtype=bfloat16_dtype),
        'delta': np.array([[0.5, -0.5]], dtype=bfloat16_dtype),
        'use_locking': False,
        'name': 'test_bfloat16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: Large dimension tensor
    input_dict_6 = {
        'var': np.ones((4, 4, 4), dtype=np.float32),
        'alpha': np.array(0.001, dtype=np.float32),
        'delta': np.random.rand(4, 4, 4).astype(np.float32),
        'use_locking': False,
        'name': 'large_tensor_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Negative alpha
    input_dict_7 = {
        'var': np.array([10.0, 20.0], dtype=np.float64),
        'alpha': np.array(-0.1, dtype=np.float64),
        'delta': np.array([1.0, 1.0], dtype=np.float64),
        'use_locking': False,
        'name': 'negative_alpha_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyGradientDescent"] = get_apply_gradient_descent_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyGradientDescent' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyGradientDescent'.")

check_valid('tf.raw_ops.ApplyGradientDescent', generated_inputs['tf.raw_ops.ApplyGradientDescent'], lib="tf", suffix=0)
