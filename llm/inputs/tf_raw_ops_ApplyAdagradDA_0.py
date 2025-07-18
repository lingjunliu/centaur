
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_adagrad_da_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyAdagradDA.
    The mutable inputs ('var', 'gradient_accumulator', 'gradient_squared_accumulator')
    are correctly defined as tf.Variable objects to prevent the eager execution error.
    Other tensor inputs are tf.constant. This is the only way to satisfy the API's
    requirement for mutable reference inputs.
    """
    list_of_inputs = []

    # --- Case 1: Basic float32, 1D ---
    input_dict_1 = {
        'var': tf.Variable([1.0, 2.0, 3.0], dtype=tf.float32),
        'gradient_accumulator': tf.Variable([0.1, 0.1, 0.1], dtype=tf.float32),
        'gradient_squared_accumulator': tf.Variable([0.01, 0.01, 0.01], dtype=tf.float32),
        'grad': tf.constant([0.5, -0.5, 0.2], dtype=tf.float32),
        'lr': tf.constant(0.001, dtype=tf.float32),
        'l1': tf.constant(1.0, dtype=tf.float32),
        'l2': tf.constant(0.5, dtype=tf.float32),
        'global_step': tf.constant(100, dtype=tf.int64),
        'use_locking': False,
        'name': 'test_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # --- Case 2: float64, 2D, use_locking=True ---
    input_dict_2 = {
        'var': tf.Variable([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64),
        'gradient_accumulator': tf.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float64),
        'gradient_squared_accumulator': tf.Variable([[0.01, 0.02], [0.03, 0.04]], dtype=tf.float64),
        'grad': tf.constant([[-0.1, 0.2], [0.3, -0.4]], dtype=tf.float64),
        'lr': tf.constant(0.01, dtype=tf.float64),
        'l1': tf.constant(0.0, dtype=tf.float64),
        'l2': tf.constant(0.1, dtype=tf.float64),
        'global_step': tf.constant(1, dtype=tf.int64),
        'use_locking': True,
        'name': 'test_float64_locking'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # --- Case 3: half (float16), 3D ---
    input_dict_3 = {
        'var': tf.Variable(np.random.randn(2, 2, 2).astype(np.float16)),
        'gradient_accumulator': tf.Variable(np.zeros((2, 2, 2), dtype=np.float16)),
        'gradient_squared_accumulator': tf.Variable(np.ones((2, 2, 2), dtype=np.float16)),
        'grad': tf.constant(np.random.randn(2, 2, 2).astype(np.float16)),
        'lr': tf.constant(0.005, dtype=tf.float16),
        'l1': tf.constant(0.2, dtype=tf.float16),
        'l2': tf.constant(0.3, dtype=tf.float16),
        'global_step': tf.constant(5000, dtype=tf.int64),
        'use_locking': False,
        'name': 'test_float16_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # --- Case 4: Scalar variables (shape=()) ---
    input_dict_4 = {
        'var': tf.Variable(1.5, dtype=tf.float32),
        'gradient_accumulator': tf.Variable(0.0, dtype=tf.float32),
        'gradient_squared_accumulator': tf.Variable(1.0, dtype=tf.float32),
        'grad': tf.constant(-0.5, dtype=tf.float32),
        'lr': tf.constant(0.01, dtype=tf.float32),
        'l1': tf.constant(0.1, dtype=tf.float32),
        'l2': tf.constant(0.2, dtype=tf.float32),
        'global_step': tf.constant(1000, dtype=tf.int64),
        'use_locking': False,
        'name': 'test_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # --- Case 5: Zero-valued inputs ---
    input_dict_5 = {
        'var': tf.Variable([0.0, 0.0], dtype=tf.float32),
        'gradient_accumulator': tf.Variable([0.0, 0.0], dtype=tf.float32),
        'gradient_squared_accumulator': tf.Variable([0.0, 0.0], dtype=tf.float32),
        'grad': tf.constant([0.0, 0.0], dtype=tf.float32),
        'lr': tf.constant(0.0, dtype=tf.float32),
        'l1': tf.constant(0.0, dtype=tf.float32),
        'l2': tf.constant(0.0, dtype=tf.float32),
        'global_step': tf.constant(0, dtype=tf.int64),
        'use_locking': False,
        'name': 'test_zeros'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyAdagradDA"] = tf_raw_ops_apply_adagrad_da_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyAdagradDA' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdagradDA'.")

check_valid('tf.raw_ops.ApplyAdagradDA', generated_inputs['tf.raw_ops.ApplyAdagradDA'], lib="tf", suffix=0)
