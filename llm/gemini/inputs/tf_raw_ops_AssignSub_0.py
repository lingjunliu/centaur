
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_assign_sub_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 vector
    input_dict = {
        'ref': np.array([10.0, 20.0], dtype=np.float32),
        'value': np.array([1.0, 2.0], dtype=np.float32),
        'use_locking': False,
        'name': 'assign_sub_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32 2D matrix
    input_dict = {
        'ref': np.array([[-10, 20], [30, -40]], dtype=np.int32),
        'value': np.array([[5, -5], [15, 1]], dtype=np.int32),
        'use_locking': False,
        'name': 'assign_sub_int32_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64 scalar
    input_dict = {
        'ref': np.array(10+5j, dtype=np.complex64),
        'value': np.array(1-2j, dtype=np.complex64),
        'use_locking': False,
        'name': 'assign_sub_complex64_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: uint8 vector
    input_dict = {
        'ref': np.array([255, 128, 10], dtype=np.uint8),
        'value': np.array([10, 28, 5], dtype=np.uint8),
        'use_locking': False,
        'name': 'assign_sub_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 vector with locking enabled
    input_dict = {
        'ref': np.array([1.23e4, 4.56e-2, 7.89], dtype=np.float64),
        'value': np.array([0.23e4, 0.56e-2, 0.89], dtype=np.float64),
        'use_locking': True,
        'name': 'assign_sub_float64_locked'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Empty tensors
    input_dict = {
        'ref': np.array([], dtype=np.float32),
        'value': np.array([], dtype=np.float32),
        'use_locking': False,
        'name': 'assign_sub_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["tf.raw_ops.AssignSub"] = tf_raw_ops_assign_sub_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AssignSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AssignSub'.")

check_valid('tf.raw_ops.AssignSub', generated_inputs['tf.raw_ops.AssignSub'], lib="tf", suffix=0)
