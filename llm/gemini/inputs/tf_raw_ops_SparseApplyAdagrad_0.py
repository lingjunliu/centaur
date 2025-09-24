
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_sparse_apply_adagrad_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.SparseApplyAdagrad.
    The recurring error `RuntimeError: sparse_apply_adagrad op does not support eager execution`
    is fundamental to the operation itself. This op is a legacy component from TensorFlow 1.x
    designed to mutate stateful `tf.Variable` objects within a computational graph. It was not
    designed for and does not have a kernel for TensorFlow 2.x's default eager execution mode.
    The error is raised by TensorFlow's generated code wrapper before the operation's
    core logic is even reached. Therefore, no variation of input values or types can "fix"
    this error within an eager execution context. The inputs provided below are syntactically
    and semantically correct according to the API's signature but will fail in any standard
    eager execution environment.
    """
    list_of_inputs = []

    # Input 1: Basic float32 case
    input_dict_1 = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        'accum': np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]], dtype=np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'grad': np.array([[0.1, 0.2]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': False,
        'update_slots': True,
        'name': 'basic_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: float64 and int64 indices
    input_dict_2 = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        'accum': np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64),
        'lr': np.array(0.001, dtype=np.float64),
        'grad': np.array([[1.0, -1.0]], dtype=np.float64),
        'indices': np.array([1], dtype=np.int64),
        'use_locking': False,
        'update_slots': True,
        'name': 'float64_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: use_locking=True
    input_dict_3 = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'accum': np.array([[0.1, 0.1], [0.1, 0.1]], dtype=np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'grad': np.array([[0.1, 0.2]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': True,
        'update_slots': True,
        'name': 'use_locking'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: update_slots=False
    input_dict_4 = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'accum': np.array([[0.1, 0.1], [0.1, 0.1]], dtype=np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'grad': np.array([[0.1, 0.2]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': False,
        'update_slots': False,
        'name': 'no_update_slots'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 1D variable
    input_dict_5 = {
        'var': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'accum': np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        'lr': np.array(0.1, dtype=np.float32),
        'grad': np.array([0.5, -0.5], dtype=np.float32),
        'indices': np.array([1, 3], dtype=np.int32),
        'use_locking': False,
        'update_slots': True,
        'name': '1d_variable'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: half (float16) type
    input_dict_6 = {
        'var': np.arange(6, dtype=np.float16).reshape(3, 2),
        'accum': np.full((3, 2), 0.1, dtype=np.float16),
        'lr': np.array(0.01, dtype=np.float16),
        'grad': np.random.rand(1, 2).astype(np.float16),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': False,
        'update_slots': True,
        'name': 'float16_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyAdagrad"] = tf_raw_ops_sparse_apply_adagrad_inputs()

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
