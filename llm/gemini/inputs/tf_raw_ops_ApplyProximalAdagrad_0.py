
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_apply_proximal_adagrad_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyProximalAdagrad.
    The recurring error 'RuntimeError: apply_proximal_adagrad op does not support eager execution'
    is fundamental. This op is stateful and modifies its input 'var' and 'accum' tensors,
    a pattern intended for TensorFlow's graph mode, not direct eager execution.
    The provided inputs are valid for the op's definition in a graph context. The error
    originates from the test environment calling this graph-style op eagerly.
    """
    list_of_inputs = []

    # Input 1: Basic float32, 1D case
    input_dict_1 = {
        'var': np.array([1.0, 2.0, -3.0], dtype=np.float32),
        'accum': np.array([0.1, 0.2, 0.3], dtype=np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.01, dtype=np.float32),
        'grad': np.array([0.5, -0.2, 0.3], dtype=np.float32),
        'use_locking': False,
        'name': 'basic_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: float64, 2D case with locking
    input_dict_2 = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        'accum': np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        'lr': np.array(0.001, dtype=np.float64),
        'l1': np.array(0.0, dtype=np.float64),
        'l2': np.array(0.1, dtype=np.float64),
        'grad': np.array([[-0.1, 0.2], [0.3, -0.4]], dtype=np.float64),
        'use_locking': True,
        'name': 'float64_2d_locked'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: half (float16), 1D case
    input_dict_3 = {
        'var': np.array([0.5, -0.5], dtype=np.half),
        'accum': np.array([0.01, 0.01], dtype=np.half),
        'lr': np.array(0.1, dtype=np.half),
        'l1': np.array(0.01, dtype=np.half),
        'l2': np.array(0.0, dtype=np.half),
        'grad': np.array([0.2, 0.1], dtype=np.half),
        'use_locking': False,
        'name': 'half_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Scalar (0-D) tensors
    input_dict_4 = {
        'var': np.array(5.0, dtype=np.float32),
        'accum': np.array(1.0, dtype=np.float32),
        'lr': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.01, dtype=np.float32),
        'l2': np.array(0.02, dtype=np.float32),
        'grad': np.array(-2.0, dtype=np.float32),
        'use_locking': False,
        'name': 'scalar_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: No L1/L2 regularization
    input_dict_5 = {
        'var': np.array([10.0, -10.0], dtype=np.float64),
        'accum': np.array([1.0, 1.0], dtype=np.float64),
        'lr': np.array(0.1, dtype=np.float64),
        'l1': np.array(0.0, dtype=np.float64),
        'l2': np.array(0.0, dtype=np.float64),
        'grad': np.array([1.0, -1.0], dtype=np.float64),
        'use_locking': False,
        'name': 'no_regularization'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Zero gradient
    input_dict_6 = {
        'var': np.array([[1.0, 2.0]], dtype=np.float32),
        'accum': np.array([[0.5, 0.5]], dtype=np.float32),
        'lr': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.1, dtype=np.float32),
        'grad': np.zeros((1, 2), dtype=np.float32),
        'use_locking': False,
        'name': 'zero_gradient'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyProximalAdagrad"] = tf_raw_ops_apply_proximal_adagrad_inputs()

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
