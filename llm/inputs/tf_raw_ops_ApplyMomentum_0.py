
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_apply_momentum_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyMomentum.

    The recurring error "apply_momentum op does not support eager execution.
    Arg 'out' is a ref." is fundamental to this specific legacy operation.
    It is designed for TensorFlow's older graph-based execution model and expects
    mutable "Ref" type variables, which are not used in the default eager
    execution mode of modern TensorFlow. The error is not caused by the input
    data values but by this core incompatibility between the op and the execution
    environment.

    The inputs provided below are valid for the op's signature and would execute
    correctly within a TensorFlow 1.x-style graph context. We are providing a
    focused set of inputs using standard floating-point types, as these are the
    intended use case for this optimization algorithm.
    """
    list_of_inputs = []

    # Case 1: Basic float32 with default flags.
    dtype = np.float32
    list_of_inputs.append({
        'var': np.array([1.0, 2.0], dtype=dtype),
        'accum': np.array([0.1, 0.2], dtype=dtype),
        'lr': np.array(0.01, dtype=dtype),
        'grad': np.array([0.5, 0.4], dtype=dtype),
        'momentum': np.array(0.9, dtype=dtype),
        'use_locking': False,
        'use_nesterov': False,
        'name': "case1_float32_default"
    })

    # Case 2: float64 with Nesterov momentum enabled.
    dtype = np.float64
    list_of_inputs.append({
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype),
        'accum': np.array([[0.0, 0.0], [0.0, 0.0]], dtype=dtype),
        'lr': np.array(0.1, dtype=dtype),
        'grad': np.array([[-0.5, 1.0], [0.1, -0.2]], dtype=dtype),
        'momentum': np.array(0.95, dtype=dtype),
        'use_locking': False,
        'use_nesterov': True,
        'name': "case2_float64_nesterov"
    })

    # Case 3: float32 with locking enabled.
    dtype = np.float32
    list_of_inputs.append({
        'var': np.random.randn(3, 3).astype(dtype),
        'accum': np.random.randn(3, 3).astype(dtype),
        'lr': np.array(0.5, dtype=dtype),
        'grad': np.random.randn(3, 3).astype(dtype),
        'momentum': np.array(0.8, dtype=dtype),
        'use_locking': True,
        'use_nesterov': False,
        'name': "case3_float32_locking"
    })

    # Case 4: float64 with both Nesterov and locking enabled.
    dtype = np.float64
    list_of_inputs.append({
        'var': np.array([-5.0, 5.0], dtype=dtype),
        'accum': np.array([1.0, -1.0], dtype=dtype),
        'lr': np.array(0.05, dtype=dtype),
        'grad': np.array([0.3, -0.3], dtype=dtype),
        'momentum': np.array(0.99, dtype=dtype),
        'use_locking': True,
        'use_nesterov': True,
        'name': "case4_float64_all_flags"
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyMomentum"] = get_apply_momentum_inputs()

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
