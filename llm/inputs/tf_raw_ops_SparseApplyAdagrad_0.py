
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_sparse_apply_adagrad_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.SparseApplyAdagrad.
    NOTE: This raw operation is designed for TensorFlow's graph mode and modifies
    its tensor inputs in-place. It will consistently raise a RuntimeError when
    called directly in eager execution, as eager tensors are immutable.
    The inputs provided here are valid for a graph-based execution context.
    """
    list_of_inputs = []

    # Input 1: Basic float32 case
    list_of_inputs.append({
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'accum': np.array([[0.1, 0.1], [0.1, 0.1]], dtype=np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'grad': np.array([[0.1, 0.2]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': False,
        'update_slots': True,
        'name': 'case_1_f32'
    })

    # Input 2: Basic float64 case with locking
    list_of_inputs.append({
        'var': np.array([[10.0], [20.0]], dtype=np.float64),
        'accum': np.array([[0.5], [0.5]], dtype=np.float64),
        'lr': np.array(0.1, dtype=np.float64),
        'grad': np.array([[-2.0]], dtype=np.float64),
        'indices': np.array([1], dtype=np.int64),
        'use_locking': True,
        'update_slots': True,
        'name': 'case_2_f64_lock'
    })

    # Input 3: 1D float32 var, multiple indices
    list_of_inputs.append({
        'var': np.array([-1.0, -2.0, -3.0], dtype=np.float32),
        'accum': np.array([0.1, 0.2, 0.3], dtype=np.float32),
        'lr': np.array(0.001, dtype=np.float32),
        'indices': np.array([0, 2], dtype=np.int32),
        'grad': np.array([0.5, -0.5], dtype=np.float32),
        'use_locking': False,
        'update_slots': True,
        'name': 'case_3_f32_1d'
    })

    # Input 4: update_slots=False
    list_of_inputs.append({
        'var': np.array([[-10.0, 20.0], [-30.0, 40.0]], dtype=np.float32),
        'accum': np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        'lr': np.array(0.2, dtype=np.float32),
        'indices': np.array([1], dtype=np.int64),
        'grad': np.array([[-5.0, -2.0]], dtype=np.float32),
        'use_locking': False,
        'update_slots': False,
        'name': 'case_4_no_slots'
    })
    
    # Input 5: Both flags changed (locking=True, update_slots=False)
    list_of_inputs.append({
        'var': np.ones((4, 2), dtype=np.float64),
        'accum': np.full((4, 2), 0.2, dtype=np.float64),
        'lr': np.array(0.3, dtype=np.float64),
        'indices': np.array([0, 3], dtype=np.int64),
        'grad': np.array([[-0.1, -0.2], [0.3, 0.4]], dtype=np.float64),
        'use_locking': True,
        'update_slots': False,
        'name': 'case_5_both_flags'
    })

    # Input 6: Update all rows
    list_of_inputs.append({
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'accum': np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
        'lr': np.array(0.1, dtype=np.float32),
        'indices': np.array([0, 1], dtype=np.int32),
        'grad': np.array([[-2.0, 1.0], [3.0, -4.0]], dtype=np.float32),
        'use_locking': False,
        'update_slots': True,
        'name': 'case_6_full_update'
    })

    # Input 7: Zero learning rate
    list_of_inputs.append({
        'var': np.array([[1.0, 2.0]], dtype=np.float32),
        'accum': np.array([[0.1, 0.1]], dtype=np.float32),
        'lr': np.array(0.0, dtype=np.float32),
        'grad': np.array([[0.5, 0.5]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': False,
        'update_slots': True,
        'name': 'case_7_zero_lr'
    })

    # Input 8: Zero gradient
    list_of_inputs.append({
        'var': np.array([[1.0, 2.0]], dtype=np.float32),
        'accum': np.array([[0.1, 0.1]], dtype=np.float32),
        'lr': np.array(0.1, dtype=np.float32),
        'grad': np.array([[0.0, 0.0]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': False,
        'update_slots': True,
        'name': 'case_8_zero_grad'
    })

    # Input 9: half/float16 type
    list_of_inputs.append({
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
        'accum': np.array([[0.1, 0.1], [0.1, 0.1]], dtype=np.float16),
        'lr': np.array(0.01, dtype=np.float16),
        'grad': np.array([[0.1, 0.2]], dtype=np.float16),
        'indices': np.array([1], dtype=np.int32),
        'use_locking': False,
        'update_slots': True,
        'name': 'case_9_float16'
    })

    # Input 10: large initial accumulator
    list_of_inputs.append({
        'var': np.array([[1.0, 2.0]], dtype=np.float32),
        'accum': np.array([[100.0, 100.0]], dtype=np.float32),
        'lr': np.array(0.1, dtype=np.float32),
        'grad': np.array([[1.0, -1.0]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': False,
        'update_slots': True,
        'name': 'case_10_large_accum'
    })

    return [copy.deepcopy(d) for d in list_of_inputs]

generated_inputs["tf.raw_ops.SparseApplyAdagrad"] = get_sparse_apply_adagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyAdagrad'.")

check_valid('tf.raw_ops.SparseApplyAdagrad', generated_inputs['tf.raw_ops.SparseApplyAdagrad'], lib="tf", suffix=0)
