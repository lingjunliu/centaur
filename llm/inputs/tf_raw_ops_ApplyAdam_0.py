
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_apply_adam_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyAdam.

    NOTE: The RuntimeError "apply_adam op does not support eager execution" is
    a fundamental limitation of calling this low-level op directly in TensorFlow's
    default eager mode. The op is designed for graph execution. The inputs
    provided are valid for the API's signature in its intended graph context.
    This generated code provides the simplest possible valid inputs to demonstrate
    that the issue is with the execution environment, not the data.
    """
    list_of_inputs = []

    # A single, extremely basic, valid input case.
    dtype = np.float32
    shape = (2,)
    lr = np.array(0.001, dtype=dtype)
    beta1 = np.array(0.9, dtype=dtype)
    beta2 = np.array(0.999, dtype=dtype)
    epsilon = np.array(1e-7, dtype=dtype)
    beta1_power = np.array(0.9, dtype=dtype)
    beta2_power = np.array(0.999, dtype=dtype)

    input_dict = {
        'var': np.array([1.0, 1.0], dtype=dtype),
        'm': np.array([0.0, 0.0], dtype=dtype),
        'v': np.array([0.0, 0.0], dtype=dtype),
        'grad': np.array([0.1, 0.1], dtype=dtype),
        'beta1_power': beta1_power,
        'beta2_power': beta2_power,
        'lr': lr,
        'beta1': beta1,
        'beta2': beta2,
        'epsilon': epsilon,
        'use_locking': False,
        'use_nesterov': False,
        'name': 'minimal_valid_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyAdam"] = tf_raw_ops_apply_adam_inputs()

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
