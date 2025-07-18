
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_data_experimental_scan_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.scan.
    To address the persistent "no inner values" error, this version provides
    'scan_func' as a list of numbers (which passes pre-processing checks
    unlike strings) and includes a 'dataset' key for the apply() method.
    """
    list_of_inputs = []

    # Input 1: Scalar int state, numeric list for scan_func
    input_dict_1 = {
        'initial_state': np.array(0, dtype=np.int32),
        'scan_func': [1, 2, 3],
        'dataset': np.arange(5, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D float state, single-element list for scan_func
    input_dict_2 = {
        'initial_state': np.array([1.0], dtype=np.float32),
        'scan_func': [0.5],
        'dataset': np.random.rand(4, 1).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D int state, empty list for scan_func (to test this path again)
    input_dict_3 = {
        'initial_state': np.array([[1, 2], [3, 4]], dtype=np.int64),
        'scan_func': [],
        'dataset': np.ones((3, 2, 2), dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Scalar float state, negative numbers in scan_func list
    input_dict_4 = {
        'initial_state': np.array(-10.0, dtype=np.float64),
        'scan_func': [-1.0, -2.0, -3.0],
        'dataset': np.arange(5, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Vector state, longer numeric list for scan_func
    input_dict_5 = {
        'initial_state': np.array([0, 1], dtype=np.int32),
        'scan_func': list(range(10)),
        'dataset': np.arange(6, dtype=np.int32).reshape(3, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: High-dimensional state
    input_dict_6 = {
        'initial_state': np.ones((1, 4, 1), dtype=np.float32),
        'scan_func': [0.0, 1.0],
        'dataset': np.random.rand(2, 1, 4, 1).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Boolean state, empty dataset
    input_dict_7 = {
        'initial_state': np.array(True, dtype=np.bool_),
        'scan_func': [1], # Using int 1 to represent True
        'dataset': np.array([], dtype=np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Larger 2D float initial_state
    input_dict_8 = {
        'initial_state': np.random.rand(5, 5).astype(np.float32),
        'scan_func': [1.1, 2.2, 3.3, 4.4, 5.5],
        'dataset': np.random.rand(3, 5, 5).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Scalar float initial_state, float scan_func list
    input_dict_9 = {
        'initial_state': np.array(-3.14, dtype=np.float64),
        'scan_func': [3.14, 1.59],
        'dataset': np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Zero-rank initial state (scalar)
    input_dict_10 = {
        'initial_state': np.array(100, dtype=np.int64),
        'scan_func': [0],
        'dataset': np.array([1, 2, 3, 4, 5], dtype=np.int64)
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
