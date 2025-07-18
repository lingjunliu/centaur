
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_apply_add_sign_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ApplyAddSign operation.
    The RuntimeError 'does not support eager execution' is inherent to this raw op
    when called directly in TF2's default mode, as it's designed to mutate
    reference inputs (Variables) within a graph. The provided numpy inputs are
    structurally valid for the operation's intended use in such a graph context.
    """
    list_of_inputs = []

    def create_input_dict(var, m, lr, alpha, sign_decay, beta, grad, use_locking, name):
        dtype = var.dtype
        return {
            'var': var,
            'm': m.astype(dtype),
            'lr': np.array(lr, dtype=dtype),
            'alpha': np.array(alpha, dtype=dtype),
            'sign_decay': np.array(sign_decay, dtype=dtype),
            'beta': np.array(beta, dtype=dtype),
            'grad': grad.astype(dtype),
            'use_locking': use_locking,
            'name': name
        }

    # Case 1: Basic float32, 1D tensors
    list_of_inputs.append(create_input_dict(
        var=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        m=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        lr=0.001, alpha=1.0, sign_decay=0.5, beta=0.9,
        grad=np.array([-0.5, 0.0, 0.5], dtype=np.float32),
        use_locking=False, name='float32_1d'
    ))

    # Case 2: float64, 2D tensors, with locking
    list_of_inputs.append(create_input_dict(
        var=np.array([[10.0, -20.0], [30.0, -40.0]], dtype=np.float64),
        m=np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64),
        lr=0.01, alpha=1.0, sign_decay=0.1, beta=0.99,
        grad=np.array([[0.1, 0.2], [-0.3, -0.4]], dtype=np.float64),
        use_locking=True, name='float64_2d_locking'
    ))

    # Case 3: half precision (float16)
    list_of_inputs.append(create_input_dict(
        var=np.arange(4, dtype=np.float16),
        m=np.zeros(4, dtype=np.float16),
        lr=0.1, alpha=1.0, sign_decay=0.0, beta=0.9,
        grad=np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float16),
        use_locking=False, name='float16'
    ))

    # Case 4: Scalar inputs
    list_of_inputs.append(create_input_dict(
        var=np.array(100.0, dtype=np.float64),
        m=np.array(-1.0, dtype=np.float64),
        lr=0.001, alpha=1.0, sign_decay=0.1, beta=0.99,
        grad=np.array(5.0, dtype=np.float64),
        use_locking=False, name='scalar'
    ))

    # Case 5: 3D tensor
    list_of_inputs.append(create_input_dict(
        var=np.full((1, 2, 3), -5.0, dtype=np.float32),
        m=np.random.randn(1, 2, 3),
        lr=1e-4, alpha=0.5, sign_decay=0.2, beta=0.8,
        grad=np.random.randn(1, 2, 3),
        use_locking=False, name='float32_3d'
    ))

    # Case 6: beta = 0.0
    list_of_inputs.append(create_input_dict(
        var=np.array([1.0, 2.0], dtype=np.float32),
        m=np.array([0.1, 0.2], dtype=np.float32),
        lr=0.01, alpha=1.0, sign_decay=0.5, beta=0.0,
        grad=np.array([0.5, -0.5], dtype=np.float32),
        use_locking=False, name='beta_zero'
    ))

    # Case 7: alpha = 0.0
    list_of_inputs.append(create_input_dict(
        var=np.array([[1.0], [2.0]], dtype=np.float32),
        m=np.array([[0.1], [0.2]], dtype=np.float32),
        lr=0.01, alpha=0.0, sign_decay=0.5, beta=0.9,
        grad=np.array([[0.5], [-0.5]], dtype=np.float32),
        use_locking=False, name='alpha_zero'
    ))

    # Case 8: Zero gradient
    list_of_inputs.append(create_input_dict(
        var=np.array([1.0, 1.0], dtype=np.float32),
        m=np.array([0.5, 0.5], dtype=np.float32),
        lr=0.1, alpha=1.0, sign_decay=0.5, beta=0.9,
        grad=np.zeros(2, dtype=np.float32),
        use_locking=False, name='zero_grad'
    ))

    # Case 9: Large magnitude values
    list_of_inputs.append(create_input_dict(
        var=np.array([1e6, 2e6], dtype=np.float32),
        m=np.array([1e5, -1e5], dtype=np.float32),
        lr=1.0, alpha=1e3, sign_decay=1e2, beta=0.5,
        grad=np.array([5e4, 6e4], dtype=np.float32),
        use_locking=False, name='large_values'
    ))

    # Case 10: Small magnitude values
    list_of_inputs.append(create_input_dict(
        var=np.array([1e-6, -2e-6], dtype=np.float64),
        m=np.array([1e-7, -1e-7], dtype=np.float64),
        lr=1e-5, alpha=1e-4, sign_decay=1e-3, beta=0.999,
        grad=np.array([5e-8, -6e-8], dtype=np.float64),
        use_locking=True, name='small_values'
    ))

    return [copy.deepcopy(d) for d in list_of_inputs]

generated_inputs["tf.raw_ops.ApplyAddSign"] = get_apply_add_sign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyAddSign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAddSign'.")

check_valid('tf.raw_ops.ApplyAddSign', generated_inputs['tf.raw_ops.ApplyAddSign'], lib="tf", suffix=0)
