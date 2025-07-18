
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_apply_proximal_adagrad_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyProximalAdagrad.
    The op is a ref-op and is known to have issues with eager execution.
    This function provides simple, canonical inputs using standard float types
    as numpy arrays, adhering to the required signature. This is a best-effort
    attempt to generate inputs that might work in a specialized execution
    environment that can handle ref-ops.
    """
    list_of_inputs = []

    # Helper to create an input dict.
    def create_input_dict(var, accum, lr, l1, l2, grad, use_locking, name, dtype):
        # All inputs must be numpy arrays.
        # Ensure accum is strictly positive to avoid division by zero in 1/sqrt(accum).
        # Ensure lr, l1, l2 are non-negative as is standard.
        return {
            'var': np.array(var, dtype=dtype),
            'accum': np.array(accum, dtype=dtype),
            'lr': np.array(lr, dtype=dtype),
            'l1': np.array(l1, dtype=dtype),
            'l2': np.array(l2, dtype=dtype),
            'grad': np.array(grad, dtype=dtype),
            'use_locking': use_locking,
            'name': name
        }

    # --- Sticking to float32 and float64 as they are most stable for such ops ---

    # Input 1: Basic float32 case, 1D tensor. No regularization.
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(var=[1.0, 2.0], accum=[0.1, 0.2], lr=0.01, l1=0.0, l2=0.0,
                          grad=[0.5, -0.5], use_locking=False, name='test1_f32_1d', dtype=np.float32)
    ))

    # Input 2: Basic float64 case, 1D tensor, with L1/L2 regularization and locking.
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(var=[10.0, -10.0], accum=[1.0, 1.0], lr=0.1, l1=0.01, l2=0.02,
                          grad=[0.2, 0.3], use_locking=True, name='test2_f64_1d_reg', dtype=np.float64)
    ))

    # Input 3: float32 with a 2D shape (2x2).
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(var=[[1.0, 2.0], [3.0, 4.0]], accum=[[0.5, 0.5], [0.5, 0.5]], lr=0.1, l1=0.0, l2=0.0,
                          grad=[[0.1, -0.1], [0.2, -0.2]], use_locking=False, name='test3_f32_2d', dtype=np.float32)
    ))

    # Input 4: float64 with a different 2D shape (1x3).
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(var=[[10.0, 20.0, 30.0]], accum=[[1.0, 2.0, 3.0]], lr=0.5, l1=0.1, l2=0.2,
                          grad=[[-0.5, 0.5, -0.5]], use_locking=True, name='test4_f64_2d', dtype=np.float64)
    ))

    # Input 5: float32 with zero gradients to test no-op update for var.
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(var=[5.0, -5.0], accum=[1.0, 1.0], lr=0.001, l1=0.1, l2=0.2,
                          grad=[0.0, 0.0], use_locking=False, name='test5_f32_zerograd', dtype=np.float32)
    ))

    # Input 6: float32 scalar case (0-D tensor).
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(var=5.0, accum=1.0, lr=0.1, l1=0.01, l2=0.02,
                          grad=-2.0, use_locking=False, name='test6_f32_scalar', dtype=np.float32)
    ))

    # Input 7: float64 scalar case with locking.
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(var=-5.0, accum=2.0, lr=0.2, l1=0.0, l2=0.0,
                          grad=3.0, use_locking=True, name='test7_f64_scalar', dtype=np.float64)
    ))

    # Input 8: Another float32 case with different positive values.
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(var=[100., 200.], accum=[50., 50.], lr=1.0, l1=0.5, l2=0.5,
                          grad=[10., 20.], use_locking=False, name='test8_f32_positive', dtype=np.float32)
    ))
    
    # Input 9: High-dimensional float32 tensor
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(var=[[[1.0, 2.0], [3.0, 4.0]]], accum=[[[0.1, 0.1], [0.2, 0.2]]], lr=0.01, l1=0.0, l2=0.0,
                          grad=[[[0.5, -0.5], [0.1, -0.2]]], use_locking=False, name='test9_f32_3d', dtype=np.float32)
    ))

    # Input 10: half (float16) type.
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(var=[1.0, 2.0], accum=[0.1, 0.1], lr=0.01, l1=0.0, l2=0.0,
                          grad=[0.5, -0.5], use_locking=False, name='test10_half', dtype=np.float16)
    ))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyProximalAdagrad"] = get_apply_proximal_adagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyProximalAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyProximalAdagrad'.")

check_valid('tf.raw_ops.ApplyProximalAdagrad', generated_inputs['tf.raw_ops.ApplyProximalAdagrad'], lib="tf", suffix=0)
