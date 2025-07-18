
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_assign_inputs():
    list_of_inputs = []

    # The error `RuntimeError: assign op does not support eager execution` is
    # fundamental to `tf.raw_ops.Assign`. This operation is a legacy component
    # from TensorFlow's graph-based execution model and expects a "ref" tensor,
    # which is incompatible with the default eager execution mode in modern
    # TensorFlow. No combination of numpy inputs can resolve this incompatibility
    # in an eager context. The following inputs are generated to be syntactically
    # correct according to the API signature and the prompt's constraints,
    # but the runtime error is expected due to the nature of the operation itself.

    # Input 1: Basic 2D float assignment
    input_dict_1 = {
        'ref': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'value': np.array([[0.0, -1.0], [-2.0, -3.0]], dtype=np.float32),
        'validate_shape': True,
        'use_locking': True,
        'name': 'assign_2d_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D integer assignment without locking
    input_dict_2 = {
        'ref': np.array([100, 200, 300], dtype=np.int32),
        'value': np.array([1, 2, 3], dtype=np.int32),
        'validate_shape': True,
        'use_locking': False,
        'name': 'assign_1d_int_no_lock'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Scalar int64 assignment
    input_dict_3 = {
        'ref': np.array(999, dtype=np.int64),
        'value': np.array(-999, dtype=np.int64),
        'validate_shape': True,
        'use_locking': True,
        'name': 'assign_scalar_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Shape change with validate_shape=False
    input_dict_4 = {
        'ref': np.array([0.0], dtype=np.float32),
        'value': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'validate_shape': False,
        'use_locking': True,
        'name': 'assign_with_shape_change'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Boolean tensor assignment
    input_dict_5 = {
        'ref': np.array([[True], [False]], dtype=np.bool_),
        'value': np.array([[False], [True]], dtype=np.bool_),
        'validate_shape': True,
        'use_locking': True,
        'name': 'assign_boolean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Float16 tensor assignment
    input_dict_6 = {
        'ref': np.ones((2, 3), dtype=np.float16),
        'value': np.zeros((2, 3), dtype=np.float16),
        'validate_shape': True,
        'use_locking': False,
        'name': 'assign_float16_no_lock'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Complex128 tensor assignment
    input_dict_7 = {
        'ref': np.array([1+2j, 3+4j], dtype=np.complex128),
        'value': np.array([5+6j, 7+8j], dtype=np.complex128),
        'validate_shape': True,
        'use_locking': True,
        'name': 'assign_complex128'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 3D uint16 tensor assignment
    input_dict_8 = {
        'ref': np.full((2, 2, 2), 65535, dtype=np.uint16),
        'value': np.zeros((2, 2, 2), dtype=np.uint16),
        'validate_shape': True,
        'use_locking': True,
        'name': 'assign_uint16_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Empty tensor with non-zero dimensions
    input_dict_9 = {
        'ref': np.empty((0, 3), dtype=np.float32),
        'value': np.empty((0, 3), dtype=np.float32),
        'validate_shape': True,
        'use_locking': True,
        'name': 'assign_empty_3_cols'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: High-rank (4D) tensor
    input_dict_10 = {
        'ref': np.ones((1, 1, 2, 2), dtype=np.int8),
        'value': np.full((1, 1, 2, 2), -1, dtype=np.int8),
        'validate_shape': True,
        'use_locking': False,
        'name': 'assign_4d_int8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
