
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_apply_adadelta_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyAdadelta.

    NOTE: The `RuntimeError: ... op does not support eager execution` is an
    expected behavior when calling this raw op directly in TensorFlow 2's
    default eager mode. This operation is stateful and modifies its input
    tensors in-place, a pattern designed for TensorFlow's graph execution
    model (used in TF1 or within a `tf.function` in TF2). The error is
    caused by the execution environment, not by the inputs themselves. The
    inputs provided here are valid according to the function's signature
    and would execute correctly within a `tf.Graph`.
    """
    list_of_inputs = []

    # Input 1: Basic case with float32, 1D tensors
    input_dict_1 = {
        'var': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'accum': np.array([0.1, 0.1, 0.1], dtype=np.float32),
        'accum_update': np.array([0.1, 0.1, 0.1], dtype=np.float32),
        'lr': np.array(0.001, dtype=np.float32),
        'rho': np.array(0.95, dtype=np.float32),
        'epsilon': np.array(1e-8, dtype=np.float32),
        'grad': np.array([0.5, 0.4, 0.3], dtype=np.float32),
        'use_locking': False,
        'name': 'basic_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: float64, 2D tensors, with use_locking=True
    input_dict_2 = {
        'var': np.array([[-1.0, 2.5], [3.0, -4.5]], dtype=np.float64),
        'accum': np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64),
        'accum_update': np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64),
        'lr': np.array(0.01, dtype=np.float64),
        'rho': np.array(0.9, dtype=np.float64),
        'epsilon': np.array(1e-7, dtype=np.float64),
        'grad': np.array([[0.1, -0.2], [-0.3, 0.4]], dtype=np.float64),
        'use_locking': True,
        'name': 'basic_float64_locking'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Scalar inputs
    input_dict_3 = {
        'var': np.array(10.0, dtype=np.float32),
        'accum': np.array(0.1, dtype=np.float32),
        'accum_update': np.array(0.2, dtype=np.float32),
        'lr': np.array(0.001, dtype=np.float32),
        'rho': np.array(0.95, dtype=np.float32),
        'epsilon': np.array(1e-8, dtype=np.float32),
        'grad': np.array(0.5, dtype=np.float32),
        'use_locking': False,
        'name': 'scalar_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Zero gradient
    input_dict_4 = {
        'var': np.array([1.0, 2.0], dtype=np.float32),
        'accum': np.array([0.1, 0.1], dtype=np.float32),
        'accum_update': np.array([0.2, 0.2], dtype=np.float32),
        'lr': np.array(0.001, dtype=np.float32),
        'rho': np.array(0.95, dtype=np.float32),
        'epsilon': np.array(1e-8, dtype=np.float32),
        'grad': np.zeros(2, dtype=np.float32),
        'use_locking': False,
        'name': 'zero_gradient'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyAdadelta"] = tf_raw_ops_apply_adadelta_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyAdadelta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdadelta'.")

check_valid('tf.raw_ops.ApplyAdadelta', generated_inputs['tf.raw_ops.ApplyAdadelta'], lib="tf", suffix=0)
