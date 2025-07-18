
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_apply_proximal_gradient_descent_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyProximalGradientDescent.
    This op is designed for graph mode and will raise a RuntimeError in eager execution.
    The provided inputs are valid for the op's signature when used in a graph context.
    """
    list_of_inputs = []

    # Case 1: Basic float32, 1D
    list_of_inputs.append(copy.deepcopy({
        'var': np.array([1.0, -2.0, 3.0], dtype=np.float32),
        'alpha': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.2, dtype=np.float32),
        'l2': np.array(0.01, dtype=np.float32),
        'delta': np.array([0.5, 1.0, -0.5], dtype=np.float32),
        'use_locking': False,
        'name': 'f32_1d'
    }))

    # Case 2: Basic float64, 2D
    list_of_inputs.append(copy.deepcopy({
        'var': np.array([[1.0, 2.0], [-3.0, -4.0]], dtype=np.float64),
        'alpha': np.array(0.05, dtype=np.float64),
        'l1': np.array(0.1, dtype=np.float64),
        'l2': np.array(0.2, dtype=np.float64),
        'delta': np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float64),
        'use_locking': False,
        'name': 'f64_2d'
    }))

    # Case 3: float32 with locking enabled
    list_of_inputs.append(copy.deepcopy({
        'var': np.array([5.0, 5.0], dtype=np.float32),
        'alpha': np.array(0.2, dtype=np.float32),
        'l1': np.array(0.3, dtype=np.float32),
        'l2': np.array(0.4, dtype=np.float32),
        'delta': np.array([-1.0, 1.0], dtype=np.float32),
        'use_locking': True,
        'name': 'f32_locking'
    }))

    # Case 4: float64 with no L1 regularization
    list_of_inputs.append(copy.deepcopy({
        'var': np.array([10., -10.], dtype=np.float64),
        'alpha': np.array(0.5, dtype=np.float64),
        'l1': np.array(0.0, dtype=np.float64),
        'l2': np.array(0.1, dtype=np.float64),
        'delta': np.random.randn(2).astype(np.float64),
        'use_locking': False,
        'name': 'f64_no_l1'
    }))

    # Case 5: float32 with no L2 regularization
    list_of_inputs.append(copy.deepcopy({
        'var': np.array([[100.], [200.]], dtype=np.float32),
        'alpha': np.array(0.01, dtype=np.float32),
        'l1': np.array(1.5, dtype=np.float32),
        'l2': np.array(0.0, dtype=np.float32),
        'delta': np.array([[10.], [-20.]], dtype=np.float32),
        'use_locking': False,
        'name': 'f32_no_l2'
    }))

    # Case 6: float64 with no regularization at all
    list_of_inputs.append(copy.deepcopy({
        'var': np.array([[0.5, -0.5]], dtype=np.float64),
        'alpha': np.array(0.1, dtype=np.float64),
        'l1': np.array(0.0, dtype=np.float64),
        'l2': np.array(0.0, dtype=np.float64),
        'delta': np.array([[0.2, 0.3]], dtype=np.float64),
        'use_locking': False,
        'name': 'f64_no_reg'
    }))

    # Case 7: float32 scalar variable
    list_of_inputs.append(copy.deepcopy({
        'var': np.array(-50.0, dtype=np.float32),
        'alpha': np.array(0.02, dtype=np.float32),
        'l1': np.array(0.6, dtype=np.float32),
        'l2': np.array(0.3, dtype=np.float32),
        'delta': np.array(2.0, dtype=np.float32),
        'use_locking': False,
        'name': 'f32_scalar'
    }))

    # Case 8: float64 with zero delta (no gradient update)
    list_of_inputs.append(copy.deepcopy({
        'var': np.array([10.0, 20.0], dtype=np.float64),
        'alpha': np.array(0.1, dtype=np.float64),
        'l1': np.array(1.0, dtype=np.float64),
        'l2': np.array(1.0, dtype=np.float64),
        'delta': np.array([0.0, 0.0], dtype=np.float64),
        'use_locking': False,
        'name': 'f64_zero_delta'
    }))

    # Case 9: float32 with zero alpha (no learning)
    list_of_inputs.append(copy.deepcopy({
        'var': np.array([1., 2., 3.], dtype=np.float32),
        'alpha': np.array(0.0, dtype=np.float32),
        'l1': np.array(1.0, dtype=np.float32),
        'l2': np.array(1.0, dtype=np.float32),
        'delta': np.array([10., 20., 30.], dtype=np.float32),
        'use_locking': False,
        'name': 'f32_zero_alpha'
    }))

    # Case 10: float16 (half precision)
    list_of_inputs.append(copy.deepcopy({
        'var': np.array([1.0, -2.0, 3.0], dtype=np.float16),
        'alpha': np.array(0.1, dtype=np.float16),
        'l1': np.array(0.2, dtype=np.float16),
        'l2': np.array(0.01, dtype=np.float16),
        'delta': np.array([0.5, 1.0, -0.5], dtype=np.float16),
        'use_locking': False,
        'name': 'f16_basic'
    }))
    
    # Case 11: 3D float32 tensor
    list_of_inputs.append(copy.deepcopy({
        'var': np.random.randn(2, 2, 2).astype(np.float32),
        'alpha': np.array(0.01, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.1, dtype=np.float32),
        'delta': np.random.randn(2, 2, 2).astype(np.float32),
        'use_locking': False,
        'name': 'f32_3d'
    }))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyProximalGradientDescent"] = tf_raw_ops_apply_proximal_gradient_descent_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyProximalGradientDescent' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyProximalGradientDescent'.")

check_valid('tf.raw_ops.ApplyProximalGradientDescent', generated_inputs['tf.raw_ops.ApplyProximalGradientDescent'], lib="tf", suffix=0)
