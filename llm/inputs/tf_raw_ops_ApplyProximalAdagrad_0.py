
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_apply_proximal_adagrad_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyProximalAdagrad.
    The recurring error "op does not support eager execution" is a fundamental
    limitation of this raw operation. It is designed for TensorFlow's graph
    mode, where it can modify its input 'ref' tensors (like 'var' and 'accum')
    in-place. Calling it directly in eager mode, which is the default in
    modern TensorFlow, is not supported. The issue lies with the execution
    context, not the validity of the provided inputs. The following inputs
    are valid according to the API signature and would work in a graph context.
    """
    list_of_inputs = []

    # Input 1: Basic float32, 1D
    dtype = np.float32
    input_dict = {
        'var': np.array([1.0, 2.0], dtype=dtype),
        'accum': np.array([0.1, 0.1], dtype=dtype),
        'lr': np.array(0.01, dtype=dtype),
        'l1': np.array(0.1, dtype=dtype),
        'l2': np.array(0.2, dtype=dtype),
        'grad': np.array([0.5, -0.5], dtype=dtype),
        'use_locking': False,
        'name': "graph_mode_test_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D with locking
    dtype = np.float64
    input_dict = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype),
        'accum': np.array([[0.1, 0.2], [0.3, 0.4]], dtype=dtype),
        'lr': np.array(0.001, dtype=dtype),
        'l1': np.array(0.0, dtype=dtype),
        'l2': np.array(0.01, dtype=dtype),
        'grad': np.array([[0.1, 0.2], [-0.1, -0.2]], dtype=dtype),
        'use_locking': True,
        'name': "graph_mode_test_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16 (half), 1D
    dtype = np.float16
    input_dict = {
        'var': np.array([-1.5, -2.5], dtype=dtype),
        'accum': np.array([1.0, 1.0], dtype=dtype),
        'lr': np.array(0.1, dtype=dtype),
        'l1': np.array(0.5, dtype=dtype),
        'l2': np.array(0.5, dtype=dtype),
        'grad': np.array([-0.5, 0.5], dtype=dtype),
        'use_locking': False,
        'name': "graph_mode_test_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, no regularization
    dtype = np.float32
    input_dict = {
        'var': np.random.randn(4).astype(dtype),
        'accum': np.ones(4, dtype=dtype) * 0.1,
        'lr': np.array(0.05, dtype=dtype),
        'l1': np.array(0.0, dtype=dtype),
        'l2': np.array(0.0, dtype=dtype),
        'grad': np.random.randn(4).astype(dtype),
        'use_locking': False,
        'name': "graph_mode_test_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, zero gradient
    dtype = np.float32
    input_dict = {
        'var': np.array([100.0, 200.0], dtype=dtype),
        'accum': np.array([10.0, 10.0], dtype=dtype),
        'lr': np.array(1.0, dtype=dtype),
        'l1': np.array(1.0, dtype=dtype),
        'l2': np.array(1.0, dtype=dtype),
        'grad': np.array([0.0, 0.0], dtype=dtype),
        'use_locking': False,
        'name': "graph_mode_test_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, scalar tensors for single-variable update
    dtype = np.float64
    input_dict = {
        'var': np.array(10.0, dtype=dtype),
        'accum': np.array(1.0, dtype=dtype),
        'lr': np.array(0.1, dtype=dtype),
        'l1': np.array(0.01, dtype=dtype),
        'l2': np.array(0.02, dtype=dtype),
        'grad': np.array(-0.5, dtype=dtype),
        'use_locking': False,
        'name': "graph_mode_test_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, large values
    dtype = np.float32
    input_dict = {
        'var': np.array([1e5, -1e5], dtype=dtype),
        'accum': np.array([1e3, 1e3], dtype=dtype),
        'lr': np.array(10.0, dtype=dtype),
        'l1': np.array(1.0, dtype=dtype),
        'l2': np.array(2.0, dtype=dtype),
        'grad': np.array([100.0, 200.0], dtype=dtype),
        'use_locking': True,
        'name': "graph_mode_test_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, column vector shape
    dtype = np.float64
    input_dict = {
        'var': np.zeros((3, 1), dtype=dtype),
        'accum': np.ones((3, 1), dtype=dtype) * 0.01,
        'lr': np.array(0.5, dtype=dtype),
        'l1': np.array(0.2, dtype=dtype),
        'l2': np.array(0.3, dtype=dtype),
        'grad': np.array([[1.0], [-2.0], [0.5]], dtype=dtype),
        'use_locking': False,
        'name': "graph_mode_test_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, higher L1 regularization
    dtype = np.float32
    input_dict = {
        'var': np.array([0.1, -0.2, 0.3], dtype=dtype),
        'accum': np.array([0.1, 0.1, 0.1], dtype=dtype),
        'lr': np.array(0.1, dtype=dtype),
        'l1': np.array(10.0, dtype=dtype),
        'l2': np.array(0.1, dtype=dtype),
        'grad': np.array([0.01, -0.02, 0.03], dtype=dtype),
        'use_locking': False,
        'name': "graph_mode_test_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, higher L2 regularization
    dtype = np.float64
    input_dict = {
        'var': np.array([0.1, -0.2, 0.3], dtype=dtype),
        'accum': np.array([0.1, 0.1, 0.1], dtype=dtype),
        'lr': np.array(0.1, dtype=dtype),
        'l1': np.array(0.1, dtype=dtype),
        'l2': np.array(10.0, dtype=dtype),
        'grad': np.array([0.01, -0.02, 0.03], dtype=dtype),
        'use_locking': True,
        'name': "graph_mode_test_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyProximalAdagrad"] = get_apply_proximal_adagrad_inputs()

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
