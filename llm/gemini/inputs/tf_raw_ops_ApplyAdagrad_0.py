
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_apply_adagrad_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyAdagrad.

    The recurring error `RuntimeError: apply_adagrad op does not support eager
    execution. Arg 'out' is a ref.` indicates a fundamental incompatibility between
    the requested legacy API and the likely execution environment. This API is
    designed for TensorFlow's graph mode and requires mutable `tf.Variable`
    (a "ref" tensor) inputs for `var` and `accum` to perform its in-place
    update.

    The test environment appears to run in eager mode and provides immutable
    `tf.Tensor` objects (converted from the provided numpy arrays). Passing
    an immutable tensor to an op that requires a mutable reference is not
    supported and causes this runtime error. The modern, eager-compatible
    equivalent is `tf.raw_ops.ResourceApplyAdagrad`.

    Given these constraints, it is highly probable that no input in the required
    numpy format can successfully execute with this specific legacy API in an
    eager context. The following inputs are provided as a best-effort attempt.
    They are semantically and syntactically correct according to the API's
    documentation but are expected to fail in the described test environment.
    """
    list_of_inputs = []

    def create_input_dict(var, accum, lr, grad, use_locking, update_slots, name=None):
        return {
            'var': var,
            'accum': accum,
            'lr': lr,
            'grad': grad,
            'use_locking': use_locking,
            'update_slots': update_slots,
            'name': name
        }

    # Case 1: Basic float32, 1D
    list_of_inputs.append(create_input_dict(
        var=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        accum=np.array([0.1, 0.1, 0.1], dtype=np.float32),
        lr=np.array(0.01, dtype=np.float32),
        grad=np.array([0.5, -0.5, 0.2], dtype=np.float32),
        use_locking=False, update_slots=True, name="case1_float32"
    ))

    # Case 2: float64, 2D
    list_of_inputs.append(create_input_dict(
        var=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        accum=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64),
        lr=np.array(0.1, dtype=np.float64),
        grad=np.array([[0.1, 0.2], [-0.1, -0.2]], dtype=np.float64),
        use_locking=False, update_slots=True, name="case2_float64"
    ))

    # Case 3: Scalar tensors
    list_of_inputs.append(create_input_dict(
        var=np.array(100.0, dtype=np.float32),
        accum=np.array(1.0, dtype=np.float32),
        lr=np.array(0.1, dtype=np.float32),
        grad=np.array(-5.0, dtype=np.float32),
        use_locking=False, update_slots=True, name="case3_scalar"
    ))
    
    # Case 4: update_slots = False
    list_of_inputs.append(create_input_dict(
        var=np.array([10.0], dtype=np.float32),
        accum=np.array([1.0], dtype=np.float32),
        lr=np.array(0.1, dtype=np.float32),
        grad=np.array([2.0], dtype=np.float32),
        use_locking=False, update_slots=False, name="case4_no_update_slots"
    ))

    # Case 5: use_locking = True
    list_of_inputs.append(create_input_dict(
        var=np.array([1.0, 2.0], dtype=np.float32),
        accum=np.array([0.5, 0.5], dtype=np.float32),
        lr=np.array(0.001, dtype=np.float32),
        grad=np.array([-1.0, 2.0], dtype=np.float32),
        use_locking=True, update_slots=True, name="case5_locking"
    ))

    # Case 6: Zero gradient
    list_of_inputs.append(create_input_dict(
        var=np.array([1.0, 2.0], dtype=np.float32),
        accum=np.array([0.1, 0.1], dtype=np.float32),
        lr=np.array(0.01, dtype=np.float32),
        grad=np.array([0.0, 0.0], dtype=np.float32),
        use_locking=False, update_slots=True, name="case6_zero_grad"
    ))

    # Case 7: Accumulator starts at zero
    list_of_inputs.append(create_input_dict(
        var=np.array([5.0, -5.0], dtype=np.float32),
        accum=np.array([0.0, 0.0], dtype=np.float32),
        lr=np.array(0.1, dtype=np.float32),
        grad=np.array([0.2, -0.3], dtype=np.float32),
        use_locking=False, update_slots=True, name="case7_zero_accum"
    ))

    # Case 8: float16 (half)
    list_of_inputs.append(create_input_dict(
        var=np.array([1.0], dtype=np.float16),
        accum=np.array([0.5], dtype=np.float16),
        lr=np.array(0.001, dtype=np.float16),
        grad=np.array([10.0], dtype=np.float16),
        use_locking=False, update_slots=True, name="case8_float16"
    ))

    # Case 9: bfloat16
    bfloat16_dtype = tf.bfloat16.as_numpy_dtype
    list_of_inputs.append(create_input_dict(
        var=np.array([1.0], dtype=bfloat16_dtype),
        accum=np.array([0.2], dtype=bfloat16_dtype),
        lr=np.array(0.01, dtype=bfloat16_dtype),
        grad=np.array([0.3], dtype=bfloat16_dtype),
        use_locking=False, update_slots=True, name="case9_bfloat16"
    ))

    # Case 10: 3D tensors
    list_of_inputs.append(create_input_dict(
        var=np.random.randn(2, 1, 3).astype(np.float32),
        accum=np.full((2, 1, 3), 0.1, dtype=np.float32),
        lr=np.array(0.01, dtype=np.float32),
        grad=np.random.randn(2, 1, 3).astype(np.float32),
        use_locking=False, update_slots=True, name="case10_3d"
    ))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyAdagrad"] = tf_raw_ops_apply_adagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdagrad'.")

check_valid('tf.raw_ops.ApplyAdagrad', generated_inputs['tf.raw_ops.ApplyAdagrad'], lib="tf", suffix=0)
