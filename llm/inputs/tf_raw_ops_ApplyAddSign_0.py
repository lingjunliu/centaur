
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_apply_add_sign_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ApplyAddSign operation.

    NOTE: The recurring error `RuntimeError: apply_add_sign op does not support
    eager execution` is fundamental to this specific op. It is a legacy
    TensorFlow 1.x op designed for graph execution, which works by modifying
    "ref" Tensors in-place. This behavior is incompatible with TensorFlow 2.x's
    default eager execution mode.

    No change to the input *values* can resolve this error, as the error is
    raised by a check within the op's wrapper itself. The inputs provided below
    are syntactically valid according to the op's signature and would execute
    correctly in a compatible graph context (e.g., inside a `@tf.function`
    or a TF1 `Session`). The modern, eager-compatible equivalent is
    `tf.raw_ops.ResourceApplyAddSign`.
    """
    list_of_inputs = []

    def create_input(var, m, lr, alpha, sign_decay, beta, grad, use_locking=False, name="", dtype=np.float32):
        """Helper to create an input dictionary with specified dtype."""
        return {
            'var': np.array(var, dtype=dtype),
            'm': np.array(m, dtype=dtype),
            'lr': np.array(lr, dtype=dtype),
            'alpha': np.array(alpha, dtype=dtype),
            'sign_decay': np.array(sign_decay, dtype=dtype),
            'beta': np.array(beta, dtype=dtype),
            'grad': np.array(grad, dtype=dtype),
            'use_locking': use_locking,
            'name': name
        }

    # Case 1: Basic float32, 1D vector. A standard use case.
    list_of_inputs.append(copy.deepcopy(create_input(
        var=[1.0, 2.0], m=[0.1, 0.2], lr=0.01, alpha=0.1, sign_decay=0.9, beta=0.99, grad=[0.5, -0.5],
        name="float32_1d_basic"
    )))

    # Case 2: float64, 2D matrix with locking enabled.
    list_of_inputs.append(copy.deepcopy(create_input(
        var=[[1.0, 2.0], [3.0, 4.0]], m=[[0.1, -0.1], [0.2, -0.2]], lr=0.001, alpha=1.0, sign_decay=0.5, beta=0.9,
        grad=[[0.3, 0.4], [-0.3, -0.4]], use_locking=True, name="float64_2d_locking", dtype=np.float64
    )))

    # Case 3: Scalar inputs for all tensor arguments.
    list_of_inputs.append(copy.deepcopy(create_input(
        var=10.0, m=1.0, lr=0.1, alpha=1.0, sign_decay=0.5, beta=0.9,
        grad=-2.0, name="scalar_all"
    )))

    # Case 4: Zero gradients, should not update var based on the gradient term.
    list_of_inputs.append(copy.deepcopy(create_input(
        var=[1.0, 2.0, 3.0], m=[0.1, 0.2, 0.3], lr=0.1, alpha=0.1, sign_decay=0.5, beta=0.9,
        grad=[0.0, 0.0, 0.0], name="zero_gradients"
    )))

    # Case 5: Zero beta, which means no momentum is used.
    list_of_inputs.append(copy.deepcopy(create_input(
        var=[100.0, 200.0], m=[10.0, 20.0], lr=0.02, alpha=1.0, sign_decay=0.9, beta=0.0,
        grad=[1.0, -1.0], name="zero_beta", dtype=np.float32
    )))
    
    # Case 6: Zero sign_decay, simplifying the update formula.
    list_of_inputs.append(copy.deepcopy(create_input(
        var=[-10.0, 20.0], m=[1.0, 2.0], lr=0.1, alpha=1.0, sign_decay=0.0, beta=0.9,
        grad=[-5.0, 5.0], name="zero_sign_decay"
    )))

    # Case 7: half (float16) dtype.
    list_of_inputs.append(copy.deepcopy(create_input(
        var=[1.0, -1.0], m=[0.5, 0.5], lr=0.001, alpha=0.1, sign_decay=0.9, beta=0.99,
        grad=[0.2, 0.3], name="half_type", dtype=np.float16
    )))
    
    # Case 8: 3D tensor input.
    list_of_inputs.append(copy.deepcopy(create_input(
        var=[[[1.0]]], m=[[[0.5]]], lr=0.1, alpha=0.2, sign_decay=0.8, beta=0.8,
        grad=[[[-0.1]]], name="float32_3d"
    )))

    # Case 9: High beta value for strong momentum effect.
    list_of_inputs.append(copy.deepcopy(create_input(
        var=[[1.0], [2.0]], m=[[0.9], [0.9]], lr=0.001, alpha=1.0, sign_decay=0.9, beta=0.9999,
        grad=[[0.1], [-0.1]], name="high_beta"
    )))

    # Case 10: All signs of inputs are negative where possible
    list_of_inputs.append(copy.deepcopy(create_input(
        var=[-1.0, -2.0], m=[-0.1, -0.2], lr=0.01, alpha=0.1, sign_decay=0.9, beta=0.99, grad=[-0.5, -0.5],
        name="all_negative"
    )))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyAddSign"] = tf_raw_ops_apply_add_sign_inputs()

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
