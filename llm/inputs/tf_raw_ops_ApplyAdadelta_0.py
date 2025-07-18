
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_apply_adadelta_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ApplyAdadelta function.
    
    NOTE: tf.raw_ops.ApplyAdadelta modifies its variable inputs in-place and is
    intended for use in a TensorFlow graph. Calling it directly in eager
    execution mode will result in a RuntimeError. The inputs provided here are
    valid numpy arrays as per the request, but the error stems from the
    execution context, not the inputs themselves.
    """
    list_of_inputs = []

    # Helper to create an input dictionary
    def create_input(var, lr, rho, epsilon, use_locking, name):
        dtype = var.dtype
        return {
            'var': var.copy(),
            'accum': np.full_like(var, 0.1, dtype=dtype),
            'accum_update': np.full_like(var, 0.1, dtype=dtype),
            'lr': np.array(lr, dtype=dtype),
            'rho': np.array(rho, dtype=dtype),
            'epsilon': np.array(epsilon, dtype=dtype),
            'grad': (np.random.rand(*var.shape) * 2 - 1).astype(dtype),
            'use_locking': use_locking,
            'name': name
        }

    # Case 1: Basic float32, 1D
    list_of_inputs.append(create_input(
        var=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        lr=0.001, rho=0.95, epsilon=1e-8, use_locking=False, name="case1_f32_1d"
    ))

    # Case 2: float64, 2D
    list_of_inputs.append(create_input(
        var=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        lr=0.01, rho=0.9, epsilon=1e-7, use_locking=True, name="case2_f64_2d"
    ))
    
    # Case 3: float16 (half), 3D
    list_of_inputs.append(create_input(
        var=np.random.randn(2, 2, 2).astype(np.float16),
        lr=0.001, rho=0.95, epsilon=1e-4, use_locking=False, name="case3_f16_3d"
    ))

    # Case 4: float32, scalar (0D)
    list_of_inputs.append(create_input(
        var=np.array(100.0, dtype=np.float32),
        lr=1.0, rho=0.99, epsilon=1e-6, use_locking=True, name="case4_f32_scalar"
    ))

    # Case 5: float64, large values
    list_of_inputs.append(create_input(
        var=np.array([1e6, -2e6], dtype=np.float64),
        lr=0.001, rho=0.95, epsilon=1.0, use_locking=False, name="case5_f64_large"
    ))

    # Case 6: float32, all zeros for var, accum, and grad
    var6 = np.zeros((2, 2), dtype=np.float32)
    list_of_inputs.append({
        'var': var6,
        'accum': np.zeros_like(var6),
        'accum_update': np.zeros_like(var6),
        'lr': np.array(1.0, dtype=np.float32),
        'rho': np.array(0.9, dtype=np.float32),
        'epsilon': np.array(1e-7, dtype=np.float32),
        'grad': np.zeros_like(var6),
        'use_locking': False,
        'name': "case6_f32_zeros"
    })

    # Case 7: float32, empty tensor
    list_of_inputs.append(create_input(
        var=np.empty((2, 0), dtype=np.float32),
        lr=0.1, rho=0.9, epsilon=1e-7, use_locking=False, name="case7_f32_empty"
    ))

    # Case 8: float32, negative var and grad values
    list_of_inputs.append(create_input(
        var=np.array([-1.5, 0.5, -2.5, 3.5], dtype=np.float32),
        lr=1.0, rho=0.99, epsilon=1e-6, use_locking=True, name="case8_f32_negative"
    ))
    
    # Case 9: float64, high rank (4D)
    list_of_inputs.append(create_input(
        var=np.ones((1, 2, 1, 2), dtype=np.float64),
        lr=0.1, rho=0.9, epsilon=1e-7, use_locking=False, name="case9_f64_4d"
    ))

    # Case 10: float32, another 1D case with different hyperparameters
    list_of_inputs.append(create_input(
        var=np.array([10.0, -10.0], dtype=np.float32),
        lr=1.0, rho=0.5, epsilon=1e-1, use_locking=True, name="case10_f32_hparams"
    ))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyAdadelta"] = tf_raw_ops_apply_adadelta_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyAdadelta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdadelta'.")

check_valid('tf.raw_ops.ApplyAdadelta', generated_inputs['tf.raw_ops.ApplyAdadelta'], lib="tf", suffix=0)
