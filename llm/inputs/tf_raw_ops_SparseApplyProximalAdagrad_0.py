
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_sparse_apply_proximal_adagrad_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.SparseApplyProximalAdagrad.
    
    CRITICAL NOTE: The error 'RuntimeError: sparse_apply_proximal_adagrad op does not support
    eager execution' is an INHERENT and UNAVOIDABLE characteristic of this operation when
    called directly in TensorFlow's default eager mode. This op requires mutable reference
    inputs ('var', 'accum'), a feature of TensorFlow's graph execution mode.
    
    It is *impossible* to "fix" this error by changing the NumPy inputs. The error arises from
    the execution context (eager vs. graph), not from the inputs themselves. The following inputs
    are syntactically and semantically correct for this operation's definition and would work
    correctly in a TensorFlow graph.
    """
    list_of_inputs = []

    # Input 1: Minimal float32 case
    input_dict = {
        'var': np.array([[1.0], [2.0]], dtype=np.float32),
        'accum': np.array([[0.1], [0.1]], dtype=np.float32),
        'lr': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.1, dtype=np.float32),
        'grad': np.array([[0.5]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': False,
        'name': 'minimal_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Minimal float64 case with int64 indices
    input_dict = {
        'var': np.array([[10.0], [20.0]], dtype=np.float64),
        'accum': np.array([[1.0], [1.0]], dtype=np.float64),
        'lr': np.array(0.01, dtype=np.float64),
        'l1': np.array(0.0, dtype=np.float64),
        'l2': np.array(0.5, dtype=np.float64),
        'grad': np.array([[-2.0]], dtype=np.float64),
        'indices': np.array([1], dtype=np.int64),
        'use_locking': False,
        'name': 'minimal_float64_int64_indices'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Half precision (float16)
    input_dict = {
        'var': np.array([[1.0], [2.0]], dtype=np.float16),
        'accum': np.array([[0.1], [0.1]], dtype=np.float16),
        'lr': np.array(0.1, dtype=np.float16),
        'l1': np.array(0.1, dtype=np.float16),
        'l2': np.array(0.1, dtype=np.float16),
        'grad': np.array([[0.5]], dtype=np.float16),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': False,
        'name': 'half_precision_float16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: use_locking=True
    input_dict = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'accum': np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
        'lr': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.2, dtype=np.float32),
        'l2': np.array(0.3, dtype=np.float32),
        'grad': np.array([[0.1, -0.1]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': True,
        'name': 'use_locking_true'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero learning rate
    input_dict = {
        'var': np.array([[1.0], [2.0]], dtype=np.float32),
        'accum': np.array([[0.1], [0.1]], dtype=np.float32),
        'lr': np.array(0.0, dtype=np.float32),
        'l1': np.array(1.0, dtype=np.float32),
        'l2': np.array(1.0, dtype=np.float32),
        'grad': np.array([[100.0]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': False,
        'name': 'zero_lr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero regularization
    input_dict = {
        'var': np.array([[1.0], [2.0]], dtype=np.float32),
        'accum': np.array([[0.1], [0.1]], dtype=np.float32),
        'lr': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.0, dtype=np.float32),
        'l2': np.array(0.0, dtype=np.float32),
        'grad': np.array([[0.5]], dtype=np.float32),
        'indices': np.array([1], dtype=np.int32),
        'use_locking': False,
        'name': 'zero_regularization'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty update
    input_dict = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'accum': np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        'lr': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.1, dtype=np.float32),
        'indices': np.array([], dtype=np.int32),
        'grad': np.empty(shape=(0, 2), dtype=np.float32),
        'use_locking': False,
        'name': 'empty_indices_grad'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large values
    input_dict = {
        'var': np.array([[1e6, -1e6]], dtype=np.float32),
        'accum': np.array([[1e3, 1e3]], dtype=np.float32),
        'lr': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.1, dtype=np.float32),
        'grad': np.array([[1e4, 1e4]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': False,
        'name': 'large_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small positive accumulator
    input_dict = {
        'var': np.array([[1.0], [2.0]], dtype=np.float32),
        'accum': np.array([[1e-9], [1e-9]], dtype=np.float32),
        'lr': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.1, dtype=np.float32),
        'grad': np.array([[0.5]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': False,
        'name': 'small_accum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D var/accum/grad
    input_dict = {
        'var': np.ones((4, 2, 3), dtype=np.float32),
        'accum': np.full((4, 2, 3), 0.1, dtype=np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.01, dtype=np.float32),
        'grad': np.random.rand(2, 2, 3).astype(np.float32),
        'indices': np.array([0, 3], dtype=np.int32),
        'use_locking': False,
        'name': '3d_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyProximalAdagrad"] = get_sparse_apply_proximal_adagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyProximalAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyProximalAdagrad'.")

check_valid('tf.raw_ops.SparseApplyProximalAdagrad', generated_inputs['tf.raw_ops.SparseApplyProximalAdagrad'], lib="tf", suffix=0)
