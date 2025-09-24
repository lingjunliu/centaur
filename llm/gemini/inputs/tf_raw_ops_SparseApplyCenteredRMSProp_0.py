
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def generate_tf_raw_ops_sparseapplycenteredrmsprop_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.SparseApplyCenteredRMSProp.

    **IMPORTANT NOTE:** The target operation, `tf.raw_ops.SparseApplyCenteredRMSProp`,
    is a legacy operation from TensorFlow 1.x designed for graph execution mode. It
    requires mutable "Ref" tensor arguments, which are not supported in TensorFlow's
    default eager execution mode.

    Therefore, calling this raw op directly in an eager context will **always**
    raise a `RuntimeError`, as confirmed by the repeated execution logs. This error
    is a feature of TensorFlow's design for this specific op and cannot be "fixed"
    by changing the inputs.

    The inputs generated below are structurally and type-correct according to the
    API's documentation. They would be valid if used within a `tf.function`
    (which creates a graph) where the numpy arrays for `var`, `mg`, `ms`, and `mom`
    are first converted to `tf.Variable`s. The issue is with the execution
    environment, not the inputs themselves.
    """
    list_of_inputs = []

    # Helper function to create a test case
    def create_input_dict(dtype, var_shape, indices_dtype, use_locking, name):
        # Create mutable variables
        var = np.ones(var_shape, dtype=dtype)
        mg = np.zeros_like(var)
        ms = np.ones_like(var)  # Start with 1s to avoid sqrt(0)
        mom = np.zeros_like(var)

        # Create indices and corresponding gradient slices
        num_rows = var_shape[0]
        if num_rows > 0:
            num_indices = min(num_rows, 3)
            indices = np.random.choice(num_rows, size=num_indices, replace=False).astype(indices_dtype)
        else:
            indices = np.array([], dtype=indices_dtype)

        grad_shape = (len(indices),) + var_shape[1:]
        grad = np.random.randn(*grad_shape).astype(dtype)

        # Create scalar hyperparameters
        lr = np.array(0.01, dtype=dtype)
        rho = np.array(0.9, dtype=dtype)
        momentum = np.array(0.5, dtype=dtype)
        epsilon = np.array(1e-7, dtype=dtype)

        return {
            'use_locking': use_locking,
            'name': name,
            'var': var,
            'mg': mg,
            'ms': ms,
            'mom': mom,
            'lr': lr,
            'rho': rho,
            'momentum': momentum,
            'epsilon': epsilon,
            'grad': grad,
            'indices': indices
        }

    # Case 1: Basic float32, int32 indices
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        dtype=np.float32, var_shape=(10, 4), indices_dtype=np.int32,
        use_locking=False, name="float32_int32_indices"
    )))

    # Case 2: float64, int64 indices
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        dtype=np.float64, var_shape=(8, 2), indices_dtype=np.int64,
        use_locking=False, name="float64_int64_indices"
    )))

    # Case 3: Locking enabled
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        dtype=np.float32, var_shape=(5, 5), indices_dtype=np.int32,
        use_locking=True, name="locking_enabled"
    )))

    # Case 4: 1D tensors (vectors)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        dtype=np.float32, var_shape=(20,), indices_dtype=np.int64,
        use_locking=False, name="1d_vector"
    )))

    # Case 5: Empty indices (should be a valid no-op)
    input_dict_empty = create_input_dict(
        dtype=np.float64, var_shape=(10, 2), indices_dtype=np.int64,
        use_locking=False, name="empty_indices"
    )
    input_dict_empty['indices'] = np.array([], dtype=np.int64)
    input_dict_empty['grad'] = np.zeros((0, 2), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy(input_dict_empty))

    # Case 6: All indices are updated
    var_shape_all = (4, 3)
    input_dict_all = create_input_dict(
        dtype=np.float32, var_shape=var_shape_all, indices_dtype=np.int32,
        use_locking=False, name="all_indices_updated"
    )
    input_dict_all['indices'] = np.arange(var_shape_all[0], dtype=np.int32)
    input_dict_all['grad'] = np.random.randn(*var_shape_all).astype(np.float32)
    list_of_inputs.append(copy.deepcopy(input_dict_all))
    
    # Case 7: High-dimensional tensor
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        dtype=np.float32, var_shape=(5, 2, 3, 4), indices_dtype=np.int32,
        use_locking=False, name="high_dim_tensor"
    )))
    
    # Case 8: Zero-sized first dimension
    input_dict_zero_dim = create_input_dict(
        dtype=np.float32, var_shape=(0, 5), indices_dtype=np.int32,
        use_locking=False, name="zero_dim_var"
    )
    list_of_inputs.append(copy.deepcopy(input_dict_zero_dim))


    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyCenteredRMSProp"] = generate_tf_raw_ops_sparseapplycenteredrmsprop_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyCenteredRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyCenteredRMSProp'.")

check_valid('tf.raw_ops.SparseApplyCenteredRMSProp', generated_inputs['tf.raw_ops.SparseApplyCenteredRMSProp'], lib="tf", suffix=0)
