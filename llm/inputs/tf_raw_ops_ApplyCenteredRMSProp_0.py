
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_apply_centered_rmsprop_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyCenteredRMSProp.
    This operation is designed for TensorFlow's graph mode and will raise a
    RuntimeError in eager execution, as it requires mutable reference inputs
    which are not supported for raw ops in eager mode. The generated inputs
    are valid for a graph execution context.
    """
    list_of_inputs = []

    # Input 1: Basic float32, 1D case
    dtype = np.float32
    list_of_inputs.append(copy.deepcopy({
        'use_locking': False,
        'name': 'graph_input_1',
        'var': np.array([1.0, 2.0], dtype=dtype),
        'mg': np.array([0.1, 0.1], dtype=dtype),
        'ms': np.array([1.0, 1.0], dtype=dtype),
        'mom': np.array([0.0, 0.0], dtype=dtype),
        'lr': np.array(0.001, dtype=dtype),
        'rho': np.array(0.9, dtype=dtype),
        'momentum': np.array(0.0, dtype=dtype),
        'epsilon': np.array(1e-7, dtype=dtype),
        'grad': np.array([0.1, 0.2], dtype=dtype)
    }))

    # Input 2: float64, 2D case with locking
    dtype = np.float64
    list_of_inputs.append(copy.deepcopy({
        'use_locking': True,
        'name': 'graph_input_2',
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype),
        'mg': np.array([[0.1, 0.2], [0.3, 0.4]], dtype=dtype),
        'ms': np.array([[1.0, 1.0], [1.0, 1.0]], dtype=dtype),
        'mom': np.array([[0.5, 0.5], [0.5, 0.5]], dtype=dtype),
        'lr': np.array(0.01, dtype=dtype),
        'rho': np.array(0.95, dtype=dtype),
        'momentum': np.array(0.5, dtype=dtype),
        'epsilon': np.array(1e-8, dtype=dtype),
        'grad': np.array([[0.1, -0.2], [0.3, -0.4]], dtype=dtype)
    }))

    # Input 3: float32, zero gradient
    dtype = np.float32
    list_of_inputs.append(copy.deepcopy({
        'use_locking': False,
        'name': 'graph_input_3',
        'var': np.array([10.0], dtype=dtype),
        'mg': np.array([1.0], dtype=dtype),
        'ms': np.array([10.0], dtype=dtype),
        'mom': np.array([1.0], dtype=dtype),
        'lr': np.array(0.1, dtype=dtype),
        'rho': np.array(0.9, dtype=dtype),
        'momentum': np.array(0.9, dtype=dtype),
        'epsilon': np.array(1e-7, dtype=dtype),
        'grad': np.zeros((1,), dtype=dtype)
    }))

    # Input 4: float64, high momentum
    dtype = np.float64
    list_of_inputs.append(copy.deepcopy({
        'use_locking': False,
        'name': 'graph_input_4',
        'var': np.array([5.0, -5.0], dtype=dtype),
        'mg': np.array([0.0, 0.0], dtype=dtype),
        'ms': np.array([1.0, 1.0], dtype=dtype),
        'mom': np.array([0.1, -0.1], dtype=dtype),
        'lr': np.array(0.01, dtype=dtype),
        'rho': np.array(0.8, dtype=dtype),
        'momentum': np.array(0.99, dtype=dtype),
        'epsilon': np.array(1e-7, dtype=dtype),
        'grad': np.array([0.1, -0.1], dtype=dtype)
    }))

    # Input 5: float32, 3D tensor
    dtype = np.float32
    shape = (2, 2, 1)
    list_of_inputs.append(copy.deepcopy({
        'use_locking': False,
        'name': 'graph_input_5',
        'var': np.ones(shape, dtype=dtype),
        'mg': np.zeros(shape, dtype=dtype),
        'ms': np.ones(shape, dtype=dtype),
        'mom': np.zeros(shape, dtype=dtype),
        'lr': np.array(0.001, dtype=dtype),
        'rho': np.array(0.9, dtype=dtype),
        'momentum': np.array(0.8, dtype=dtype),
        'epsilon': np.array(1e-8, dtype=dtype),
        'grad': np.random.randn(*shape).astype(dtype)
    }))

    # Input 6: float32, zero momentum
    dtype = np.float32
    list_of_inputs.append(copy.deepcopy({
        'use_locking': False,
        'name': 'graph_input_6',
        'var': np.array([1.0, 2.0], dtype=dtype),
        'mg': np.array([0.1, 0.1], dtype=dtype),
        'ms': np.array([1.0, 1.0], dtype=dtype),
        'mom': np.array([0.5, 0.5], dtype=dtype),
        'lr': np.array(0.001, dtype=dtype),
        'rho': np.array(0.9, dtype=dtype),
        'momentum': np.array(0.0, dtype=dtype),
        'epsilon': np.array(1e-7, dtype=dtype),
        'grad': np.array([0.1, 0.2], dtype=dtype)
    }))

    # Input 7: float64, larger values
    dtype = np.float64
    list_of_inputs.append(copy.deepcopy({
        'use_locking': True,
        'name': 'graph_input_7',
        'var': np.array([1e3, -2e3], dtype=dtype),
        'mg': np.array([1e1, 2e1], dtype=dtype),
        'ms': np.array([1e4, 2e4], dtype=dtype),
        'mom': np.array([1e0, -1e0], dtype=dtype),
        'lr': np.array(1.0, dtype=dtype),
        'rho': np.array(0.99, dtype=dtype),
        'momentum': np.array(0.9, dtype=dtype),
        'epsilon': np.array(1e-2, dtype=dtype),
        'grad': np.array([1e2, -2e2], dtype=dtype)
    }))

    # Input 8: float32, 4D tensor (like for CNNs)
    dtype = np.float32
    shape = (1, 2, 2, 3)
    list_of_inputs.append(copy.deepcopy({
        'use_locking': False,
        'name': 'graph_input_8',
        'var': np.random.randn(*shape).astype(dtype),
        'mg': np.zeros(shape, dtype=dtype),
        'ms': np.ones(shape, dtype=dtype),
        'mom': np.zeros(shape, dtype=dtype),
        'lr': np.array(0.01, dtype=dtype),
        'rho': np.array(0.9, dtype=dtype),
        'momentum': np.array(0.5, dtype=dtype),
        'epsilon': np.array(1e-7, dtype=dtype),
        'grad': np.random.randn(*shape).astype(dtype)
    }))

    # Input 9: float64, scalar case (1D with one element)
    dtype = np.float64
    list_of_inputs.append(copy.deepcopy({
        'use_locking': False,
        'name': 'graph_input_9',
        'var': np.array([100.0], dtype=dtype),
        'mg': np.array([1.0], dtype=dtype),
        'ms': np.array([10.0], dtype=dtype),
        'mom': np.array([0.0], dtype=dtype),
        'lr': np.array(0.1, dtype=dtype),
        'rho': np.array(0.9, dtype=dtype),
        'momentum': np.array(0.9, dtype=dtype),
        'epsilon': np.array(1e-8, dtype=dtype),
        'grad': np.array([-5.0], dtype=dtype)
    }))

    # Input 10: No name provided
    dtype = np.float32
    list_of_inputs.append(copy.deepcopy({
        'use_locking': False,
        'name': None,
        'var': np.array([1.0, 2.0, 3.0], dtype=dtype),
        'mg': np.array([0.1, 0.2, 0.3], dtype=dtype),
        'ms': np.array([1.0, 1.0, 1.0], dtype=dtype),
        'mom': np.array([0.5, 0.5, 0.5], dtype=dtype),
        'lr': np.array(0.01, dtype=dtype),
        'rho': np.array(0.9, dtype=dtype),
        'momentum': np.array(0.5, dtype=dtype),
        'epsilon': np.array(1e-7, dtype=dtype),
        'grad': np.array([0.2, -0.1, 0.3], dtype=dtype)
    }))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyCenteredRMSProp"] = get_apply_centered_rmsprop_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyCenteredRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyCenteredRMSProp'.")

check_valid('tf.raw_ops.ApplyCenteredRMSProp', generated_inputs['tf.raw_ops.ApplyCenteredRMSProp'], lib="tf", suffix=0)
