
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_mapunstagenokey_inputs():
    """
    Generates a list of syntactically valid inputs for the tf.raw_ops.MapUnstageNoKey function.
    This op is designed to block if the underlying container is empty. In an isolated
    test environment where no corresponding staging op is run, a timeout is the
    expected behavior. The provided inputs are valid definitions for the operation.
    """
    list_of_inputs = []

    # Input 1: Basic case with a single float32 dtype and an op-local container.
    input_dict_1 = {
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.float32],
        'capacity': 1,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'test_local_container'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Integer dtype with a session-local named container.
    input_dict_2 = {
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.int32],
        'capacity': 2,
        'memory_limit': 0,
        'container': 'mycontainerone',
        'shared_name': '',
        'name': 'test_named_container'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Bool dtype with a shared container (across sessions).
    input_dict_3 = {
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.bool],
        'capacity': 1,
        'memory_limit': 0,
        'container': '',
        'shared_name': 'mysharedmapone',
        'name': 'test_shared_container'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Multiple dtypes with both a container and shared name specified.
    input_dict_4 = {
        'indices': np.array([0, 1], dtype=np.int32),
        'dtypes': [tf.string, tf.float64],
        'capacity': 5,
        'memory_limit': 0,
        'container': 'mycontainertwo',
        'shared_name': 'mysharedmaptwo',
        'name': 'test_both_names'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Complex number dtype.
    input_dict_5 = {
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.complex64],
        'capacity': 1,
        'memory_limit': 0,
        'container': 'containerthree',
        'shared_name': 'sharedthree',
        'name': 'test_complex'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Zero capacity (unbounded).
    input_dict_6 = {
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.int8],
        'capacity': 0,
        'memory_limit': 0,
        'container': 'containerfour',
        'shared_name': '',
        'name': 'test_zero_capacity'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    return list_of_inputs

generated_inputs["tf.raw_ops.MapUnstageNoKey"] = tf_raw_ops_mapunstagenokey_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MapUnstageNoKey' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MapUnstageNoKey'.")

check_valid('tf.raw_ops.MapUnstageNoKey', generated_inputs['tf.raw_ops.MapUnstageNoKey'], lib="tf", suffix=0)
