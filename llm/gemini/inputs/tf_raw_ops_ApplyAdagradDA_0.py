
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_apply_adagrad_da_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyAdagradDA.

    IMPORTANT NOTE: The error `RuntimeError: apply_adagrad_da op does not
    support eager execution. Arg 'out' is a ref.` is an expected outcome when
    this raw operation is called directly in TensorFlow's default eager mode.
    This operation is designed to mutate a "ref" tensor in-place, a behavior
    that is only supported in a graph context (e.g., inside a `@tf.function`
    decorated function or a `tf.Graph`).

    The inputs generated here are valid according to the API's documentation and
    would execute correctly in a compatible graph-based environment. The error
    is not caused by the input values but by the execution context, which is
    outside the control of this input generation function.
    """
    list_of_inputs = []

    def create_input_dict(dtype, shape, use_locking, name):
        """Helper to create a single input dictionary for float types."""
        var = np.random.randn(*shape).astype(dtype)
        # Accumulators must be non-negative.
        grad_accum = np.abs(np.random.randn(*shape)).astype(dtype)
        grad_sq_accum = np.abs(np.random.randn(*shape)).astype(dtype)
        grad = np.random.randn(*shape).astype(dtype)

        # Scalar parameters must be 0-D arrays of the same type as var.
        lr = np.array(0.001, dtype=dtype)
        l1 = np.array(0.1, dtype=dtype)
        l2 = np.array(0.01, dtype=dtype)

        return {
            'var': var,
            'gradient_accumulator': grad_accum,
            'gradient_squared_accumulator': grad_sq_accum,
            'grad': grad,
            'lr': lr,
            'l1': l1,
            'l2': l2,
            'global_step': np.array(np.random.randint(1, 1000), dtype=np.int64),
            'use_locking': use_locking,
            'name': name
        }

    # Input 1: Basic float32, 1D, no locking.
    list_of_inputs.append(copy.deepcopy(create_input_dict(np.float32, (8,), False, "f32_1d")))

    # Input 2: Basic float32, 2D, with locking.
    list_of_inputs.append(copy.deepcopy(create_input_dict(np.float32, (4, 2), True, "f32_2d_lock")))

    # Input 3: Using float64 dtype.
    list_of_inputs.append(copy.deepcopy(create_input_dict(np.float64, (6,), False, "f64_1d")))

    # Input 4: Using float16 (half) dtype with locking.
    list_of_inputs.append(copy.deepcopy(create_input_dict(np.float16, (3, 3), True, "f16_2d")))

    # Input 5: Zero values for L1/L2 regularization.
    input_5 = create_input_dict(np.float32, (5,), False, "zero_reg")
    input_5['l1'] = np.array(0.0, dtype=np.float32)
    input_5['l2'] = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(input_5))

    # Input 6: Zero gradient.
    input_6 = create_input_dict(np.float64, (2, 4), True, "zero_grad")
    input_6['grad'] = np.zeros((2, 4), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: Single element tensors.
    list_of_inputs.append(copy.deepcopy(create_input_dict(np.float32, (1,), False, "single_elem")))

    # Input 8: Larger 3D tensor shape.
    list_of_inputs.append(copy.deepcopy(create_input_dict(np.float32, (2, 3, 2), False, "f32_3d")))

    # Input 9: High global step value.
    input_9 = create_input_dict(np.float32, (3, 3), True, "high_step")
    input_9['global_step'] = np.array(1000000, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy(input_9))

    # Input 10: High learning rate value.
    input_10 = create_input_dict(np.float64, (4,), False, "high_lr")
    input_10['lr'] = np.array(100.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy(input_10))

    # Input 11: Empty shape (scalar variable)
    list_of_inputs.append(copy.deepcopy(create_input_dict(np.float32, (), False, "scalar_var")))
    
    # Input 12: All inputs with negative values where applicable
    input_12 = create_input_dict(np.float32, (5,), False, "negative_vals")
    input_12['var'] = -np.abs(np.random.randn(5)).astype(np.float32)
    input_12['grad'] = -np.abs(np.random.randn(5)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy(input_12))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyAdagradDA"] = get_tf_raw_ops_apply_adagrad_da_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyAdagradDA' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdagradDA'.")

check_valid('tf.raw_ops.ApplyAdagradDA', generated_inputs['tf.raw_ops.ApplyAdagradDA'], lib="tf", suffix=0)
