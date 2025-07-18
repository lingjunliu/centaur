
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_sparse_apply_centered_rmsprop_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.SparseApplyCenteredRMSProp.
    The repeated 'RuntimeError: ... op does not support eager execution' is
    a fundamental issue. This op requires mutable tf.Variable inputs (refs)
    to modify them in-place, a concept from TensorFlow's graph mode. The testing
    environment likely uses eager mode, where numpy arrays become immutable
    tf.Tensors, causing the incompatibility. The provided inputs are valid

    per the API documentation (testing various dtypes and shapes), but will
    fail in a standard eager execution context. This attempt explores more
    exotic data types like complex and integer types, as listed in the documentation.
    """
    list_of_inputs = []

    def create_input_dict(var_dtype, var_shape, indices_dtype, indices_list, locking, name_str):
        # Create complex or real arrays based on dtype
        if np.iscomplexobj(np.zeros(1, dtype=var_dtype)):
            var = np.random.uniform(1, 2, size=var_shape).astype(var_dtype) + 1j * np.random.uniform(1, 2, size=var_shape).astype(var_dtype)
            mg = np.random.uniform(0, 1, size=var_shape).astype(var_dtype) + 1j * np.random.uniform(0, 1, size=var_shape).astype(var_dtype)
            ms = np.random.uniform(0.1, 1, size=var_shape).astype(var_dtype) # ms is real-valued
            mom = np.zeros(var_shape, dtype=var_dtype)
            grad_shape = (len(indices_list),) + var_shape[1:]
            grad = np.random.uniform(-1, 1, size=grad_shape).astype(var_dtype) + 1j * np.random.uniform(-1, 1, size=grad_shape).astype(var_dtype)
        else:
            var = np.random.uniform(1, 2, size=var_shape).astype(var_dtype)
            mg = np.random.uniform(0, 1, size=var_shape).astype(var_dtype)
            ms = np.random.uniform(0.1, 1, size=var_shape).astype(var_dtype)
            mom = np.zeros(var_shape, dtype=var_dtype)
            grad_shape = (len(indices_list),) + var_shape[1:]
            grad = np.random.uniform(-1, 1, size=grad_shape).astype(var_dtype)

        # Scalar parameters
        lr_val = 0.01
        rho_val = 0.9
        momentum_val = 0.0
        epsilon_val = 1e-7

        # For complex types, these scalars must be real-valued but of the corresponding float type
        if var_dtype == np.complex64:
            scalar_dtype = np.float32
        elif var_dtype == np.complex128:
            scalar_dtype = np.float64
        else:
            scalar_dtype = var_dtype

        lr = np.array(lr_val, dtype=scalar_dtype)
        rho = np.array(rho_val, dtype=scalar_dtype)
        momentum = np.array(momentum_val, dtype=scalar_dtype)
        epsilon = np.array(epsilon_val, dtype=scalar_dtype)

        # For integer types, the scalars should probably be float, matching grad.
        # But docs say "must have the same type as var". This is contradictory.
        # We will follow the "same type" rule strictly, even if it's strange.
        if np.issubdtype(var_dtype, np.integer):
             lr, rho, momentum, epsilon = [x.astype(var_dtype) for x in [lr,rho,momentum,epsilon]]


        indices = np.array(indices_list, dtype=indices_dtype)

        return {
            'var': var,
            'mg': mg,
            'ms': ms,
            'mom': mom,
            'lr': lr,
            'rho': rho,
            'momentum': momentum,
            'epsilon': epsilon,
            'grad': grad,
            'indices': indices,
            'use_locking': locking,
            'name': name_str
        }

    # Case 1: Standard float32
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        np.float32, (10, 4), np.int32, [1, 5, 9], False, "f32_standard"
    )))

    # Case 2: float64 with locking
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        np.float64, (8, 2), np.int64, [0, 7], True, "f64_locking"
    )))

    # Case 3: half (float16)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        np.float16, (12, 3), np.int32, [2, 6, 10], False, "f16_half"
    )))
    
    # Case 4: complex64 - new type
    # Note: ms (mean_square) should be real, but the op signature says "same type as var".
    # Assuming the op handles this internally, we create a complex ms.
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        np.complex64, (5, 5), np.int32, [1, 3], False, "complex64"
    )))

    # Case 5: complex128 - new type
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        np.complex128, (6,), np.int64, [0, 2, 4], True, "complex128_1d"
    )))

    # Case 6: int32 - new type (unusual but listed in docs)
    # The math for RMSProp doesn't make sense for integers, but we test the type support.
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        np.int32, (7, 2), np.int32, [1, 4, 6], False, "int32_unusual"
    )))

    # Case 7: 1D variable, a common use case
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        np.float32, (100,), np.int32, [10, 20, 30], False, "f32_1d"
    )))
    
    # Case 8: Empty indices list (no-op)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        np.float64, (5, 5), np.int64, [], False, "f64_empty_indices"
    )))

    # Case 9: Non-zero momentum
    input_dict_9 = create_input_dict(
        np.float32, (9, 3), np.int32, [0, 4, 8], False, "f32_with_momentum"
    )
    input_dict_9['momentum'] = np.array(0.9, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: bfloat16
    bf16_input = create_input_dict(np.float32, (8, 8), np.int32, [3, 6], False, "bfloat16")
    for key in ['var', 'mg', 'ms', 'mom', 'lr', 'rho', 'momentum', 'epsilon', 'grad']:
        bf16_input[key] = tf.cast(tf.constant(bf16_input[key]), dtype=tf.bfloat16).numpy()
    list_of_inputs.append(copy.deepcopy(bf16_input))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyCenteredRMSProp"] = tf_raw_ops_sparse_apply_centered_rmsprop_inputs()

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
