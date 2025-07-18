
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import torch
import tensorflow as tf

def tf_raw_ops_assign_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float assignment.
    input_dict_1 = {
        'ref': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'value': np.array([4.0, 5.0, 6.0], dtype=np.float32),
        'validate_shape': True,
        'use_locking': True,
        'name': 'assign_float_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D integer assignment.
    input_dict_2 = {
        'ref': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'value': np.array([[-5, -6], [-7, -8]], dtype=np.int32),
        'validate_shape': True,
        'use_locking': True,
        'name': 'assign_int_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Scalar assignment.
    input_dict_3 = {
        'ref': np.array(100, dtype=np.int64),
        'value': np.array(200, dtype=np.int64),
        'validate_shape': True,
        'use_locking': True,
        'name': 'assign_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Reshaping assignment with validate_shape=False.
    input_dict_4 = {
        'ref': np.array([1], dtype=np.float32),
        'value': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'validate_shape': False,
        'use_locking': True,
        'name': 'assign_reshape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Assignment with use_locking=False.
    input_dict_5 = {
        'ref': np.array([10.0, 20.0], dtype=np.float64),
        'value': np.array([30.0, 40.0], dtype=np.float64),
        'validate_shape': True,
        'use_locking': False,
        'name': 'assign_no_locking'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Boolean assignment
    input_dict_6 = {
        'ref': np.array([True, False], dtype=np.bool_),
        'value': np.array([False, True], dtype=np.bool_),
        'validate_shape': True,
        'use_locking': True,
        'name': 'assign_bool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Complex number assignment
    input_dict_7 = {
        'ref': np.array([1+2j, 3+4j], dtype=np.complex64),
        'value': np.array([5+6j, 7+8j], dtype=np.complex64),
        'validate_shape': True,
        'use_locking': True,
        'name': 'assign_complex'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Higher-dimensional assignment
    input_dict_8 = {
        'ref': np.ones((2, 2, 2), dtype=np.float32),
        'value': np.zeros((2, 2, 2), dtype=np.float32),
        'validate_shape': True,
        'use_locking': True,
        'name': 'assign_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    return list_of_inputs

generated_inputs["tf.raw_ops.Assign"] = tf_raw_ops_assign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Assign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Assign'.")

check_valid('tf.raw_ops.Assign', generated_inputs['tf.raw_ops.Assign'], lib="tf", suffix=0)
