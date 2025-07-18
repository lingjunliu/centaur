
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_apply_ftrl_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyFtrl.

    The recurring error "apply_ftrl op does not support eager execution" is
    a fundamental constraint of this TensorFlow raw operation. It is designed
    to mutate a stateful variable in-place within a TensorFlow graph, a behavior
    that is incompatible with the eager execution mode used by the testing environment.

    Therefore, no change to the input *values* (e.g., shape, dtype) can resolve
    this error. The failure is inherent to calling this specific op in this
    specific execution context.

    This function provides a single, valid input that conforms to the API's
    signature, using a different data type (float16) than previous attempts.
    This represents a valid input for the op, even though it is expected to
    fail when run eagerly.
    """
    list_of_inputs = []

    # A single, canonical input using float16 (half precision).
    # The input is valid for the op's signature, but the op itself is not
    # compatible with the likely execution environment.
    var_val = np.array([1.0, 2.0], dtype=np.float16)
    # Accumulator must be non-negative.
    accum_val = np.array([0.1, 0.1], dtype=np.float16)
    linear_val = np.array([0.5, -0.5], dtype=np.float16)
    grad_val = np.array([0.2, 0.3], dtype=np.float16)
    lr_val = np.array(0.01, dtype=np.float16)
    l1_val = np.array(0.1, dtype=np.float16)
    l2_val = np.array(0.0, dtype=np.float16)
    lr_power_val = np.array(-0.5, dtype=np.float16)

    input_dict = {
        'var': var_val,
        'accum': accum_val,
        'linear': linear_val,
        'grad': grad_val,
        'lr': lr_val,
        'l1': l1_val,
        'l2': l2_val,
        'lr_power': lr_power_val,
        'use_locking': False,
        'multiply_linear_by_lr': False,
        'name': "canonical_ftrl_float16_input"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyFtrl"] = get_apply_ftrl_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyFtrl' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyFtrl'.")

check_valid('tf.raw_ops.ApplyFtrl', generated_inputs['tf.raw_ops.ApplyFtrl'], lib="tf", suffix=0)
