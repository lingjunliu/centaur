
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseApplyAdagradDA_inputs():
    # The error "sparse_apply_adagrad_da op does not support eager execution"
    # is fundamental to the op's design. It's a legacy TF1-style op that
    # mutates its inputs (ref arguments) and is incompatible with TF2's default
    # eager execution model. No change to the inputs can fix this; the execution
    # environment itself must be a TF1-style graph session.
    # The following provides a single, valid input configuration for such an
    # environment to demonstrate that the inputs themselves are not the issue.
    list_of_inputs = []

    input_dict = {
        'var': np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        'gradient_accumulator': np.full((3, 2), 0.1, dtype=np.float32),
        'gradient_squared_accumulator': np.full((3, 2), 0.01, dtype=np.float32),
        'grad': np.array([[0.1, 0.2]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'lr': np.array(0.01, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.01, dtype=np.float32),
        'global_step': np.array(100, dtype=np.int64),
        'use_locking': False,
        'name': 'graph_mode_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyAdagradDA"] = tf_raw_ops_SparseApplyAdagradDA_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyAdagradDA' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyAdagradDA'.")

check_valid('tf.raw_ops.SparseApplyAdagradDA', generated_inputs['tf.raw_ops.SparseApplyAdagradDA'], lib="tf", suffix=0)
