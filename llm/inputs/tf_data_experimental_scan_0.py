
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_scan_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.scan function.
    This API returns a function, which is then applied to a dataset.
    The 'data' key in the input dictionary represents the numpy data that will
    be converted to a tf.data.Dataset for the application.
    The 'scan_func' is provided as a list of numbers to satisfy the type
    signature constraint.
    """
    list_of_inputs = []

    # Input 1: Simple 0-D integer state, 1D integer dataset
    input_dict_1 = {
        'initial_state': np.array(0, dtype=np.int32),
        'scan_func': [1],
        'data': np.array([1, 2, 3, 4, 5], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1-D float32 state, 2D float32 dataset
    input_dict_2 = {
        'initial_state': np.array([0.0, 0.0], dtype=np.float32),
        'scan_func': [1, 2],
        'data': np.array([[1.0, -1.0], [2.0, -2.0], [3.5, -3.5]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Scalar int32 state, 3D int32 dataset
    input_dict_3 = {
        'initial_state': np.array(-10, dtype=np.int32),
        'scan_func': [],
        'data': np.arange(24, dtype=np.int32).reshape(4, 3, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Scalar float64 state, 1D float64 dataset
    input_dict_4 = {
        'initial_state': np.array(0.0, dtype=np.float64),
        'scan_func': [1, 2, 3],
        'data': np.random.rand(10).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 1-D int64 state, 2D int64 dataset
    input_dict_5 = {
        'initial_state': np.array([0, 0], dtype=np.int64),
        'scan_func': [100],
        'data': np.array([[10000000000, 2], [3, 40000000000]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 0-D float state (negative), 1D float dataset
    input_dict_6 = {
        'initial_state': np.array(-3.14, dtype=np.float32),
        'scan_func': [0],
        'data': np.array([-1.0, 0.0, 1.0, 2.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: State with complex numbers (complex64), 2D complex dataset
    input_dict_7 = {
        'initial_state': np.array(0+0j, dtype=np.complex64),
        'scan_func': [1, 2, 3],
        'data': (np.random.rand(4, 3) + 1j * np.random.rand(4, 3)).astype(np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: State with boolean values, 2D boolean dataset
    input_dict_8 = {
        'initial_state': np.array(False, dtype=np.bool_),
        'scan_func': [0, 1],
        'data': np.array([[True, True], [False, True], [False, False]], dtype=np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Non-scalar state, matching dataset elements
    input_dict_9 = {
        'initial_state': np.array([0, 0, 0], dtype=np.int16),
        'scan_func': [42],
        'data': np.arange(15, dtype=np.int16).reshape(5, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: State with complex128 dtype, matching dataset
    input_dict_10 = {
        'initial_state': np.array([0j, 0j], dtype=np.complex128),
        'scan_func': [-1],
        'data': (np.random.rand(5, 2) + 1j*np.random.rand(5,2)).astype(np.complex128)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.data.experimental.scan"] = tf_data_experimental_scan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.scan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.scan'.")

check_valid('tf.data.experimental.scan', generated_inputs['tf.data.experimental.scan'], lib="tf", suffix=0)
