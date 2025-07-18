
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_apply_adam_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyAdam.
    Note: This raw op is designed for graph execution and is not compatible
    with TensorFlow's eager execution mode. Calling it in an eager context
    is expected to raise a RuntimeError, as it attempts to modify a
    variable by reference, which is a graph-mode behavior. The generated
    inputs are valid for the op's signature in its intended graph context.
    """
    list_of_inputs = []

    def create_input_set(dtype, shape, use_locking, use_nesterov, name_suffix):
        # Tensors that share the main shape
        if np.issubdtype(dtype, np.integer):
            var = np.random.randint(-100, 100, size=shape, dtype=dtype)
            m = np.random.randint(-100, 100, size=shape, dtype=dtype)
            v = np.random.randint(0, 100, size=shape, dtype=dtype)
            grad = np.random.randint(-100, 100, size=shape, dtype=dtype)
        else:
            var = np.random.standard_normal(size=shape).astype(dtype)
            m = np.random.standard_normal(size=shape).astype(dtype)
            # v (second moment) must be non-negative.
            v = np.abs(np.random.standard_normal(size=shape)).astype(dtype)
            grad = np.random.standard_normal(size=shape).astype(dtype)

        # Scalar parameters must be 0-D numpy arrays with the same dtype.
        beta1_power = np.array(0.9**5, dtype=dtype)
        beta2_power = np.array(0.999**5, dtype=dtype)
        lr = np.array(0.001, dtype=dtype)
        beta1 = np.array(0.9, dtype=dtype)
        beta2 = np.array(0.999, dtype=dtype)
        epsilon = np.array(1e-7, dtype=dtype)

        return {
            'use_locking': use_locking,
            'use_nesterov': use_nesterov,
            'name': f'ApplyAdam_{name_suffix}',
            'var': var,
            'm': m,
            'v': v,
            'beta1_power': beta1_power,
            'beta2_power': beta2_power,
            'lr': lr,
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
            'grad': grad
        }

    # Case 1: Standard float32, 1D tensor.
    list_of_inputs.append(copy.deepcopy(create_input_set(np.float32, (10,), False, False, "f32_1d")))

    # Case 2: Standard float64, 2D tensor.
    list_of_inputs.append(copy.deepcopy(create_input_set(np.float64, (5, 5), False, False, "f64_2d")))

    # Case 3: Nesterov enabled, float32, 3D tensor.
    list_of_inputs.append(copy.deepcopy(create_input_set(np.float32, (2, 3, 4), False, True, "f32_3d_nesterov")))

    # Case 4: Locking enabled, float64, 2D tensor.
    list_of_inputs.append(copy.deepcopy(create_input_set(np.float64, (4, 2), True, False, "f64_2d_lock")))

    # Case 5: Both locking and Nesterov enabled, float32.
    list_of_inputs.append(copy.deepcopy(create_input_set(np.float32, (8,), True, True, "f32_1d_both")))

    # Case 6: Zero gradient, float64.
    input_6 = create_input_set(np.float64, (12,), False, False, "f64_zero_grad")
    input_6['grad'] = np.zeros_like(input_6['grad'])
    list_of_inputs.append(copy.deepcopy(input_6))

    # Case 7: Zero initial moments (m and v), float32.
    input_7 = create_input_set(np.float32, (3, 3), False, False, "f32_zero_moments")
    input_7['m'] = np.zeros_like(input_7['m'])
    input_7['v'] = np.zeros_like(input_7['v'])
    list_of_inputs.append(copy.deepcopy(input_7))

    # Case 8: int32 type, as allowed by docs. This is an edge case.
    list_of_inputs.append(copy.deepcopy(create_input_set(np.int32, (7,), False, False, "i32_1d")))

    # Case 9: int64 type, as allowed by docs, with Nesterov.
    list_of_inputs.append(copy.deepcopy(create_input_set(np.int64, (2, 4), False, True, "i64_2d_nesterov")))
    
    # Case 10: float16 (half) type
    list_of_inputs.append(copy.deepcopy(create_input_set(np.float16, (16,), False, False, "f16_1d")))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyAdam"] = get_apply_adam_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyAdam' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdam'.")

check_valid('tf.raw_ops.ApplyAdam', generated_inputs['tf.raw_ops.ApplyAdam'], lib="tf", suffix=0)
