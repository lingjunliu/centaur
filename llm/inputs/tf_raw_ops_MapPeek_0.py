
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_mappeek_inputs():
    """
    Generates a list of syntactically valid inputs for tf.raw_ops.MapPeek.
    NOTE: This operation is expected to time out in an isolated test environment
    because it blocks until a key is inserted into the map, which does not happen
    during the test. The inputs are valid, but the environment cannot handle
    this blocking behavior.
    """
    list_of_inputs = []

    # Input 1: Basic case. This will block and time out.
    input_dict_1 = {
        'key': np.array(6001, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.float32],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_timeout_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Using a non-default container and shared_name. Will also time out.
    input_dict_2 = {
        'key': np.array(6002, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.string],
        'capacity': 0,
        'memory_limit': 0,
        'container': 'test_container',
        'shared_name': 'test_shared_name',
        'name': 'peek_timeout_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Using non-default capacity and memory_limit. Will also time out.
    input_dict_3 = {
        'key': np.array(6003, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.int64],
        'capacity': 10,
        'memory_limit': 1024,
        'container': '',
        'shared_name': '',
        'name': 'peek_timeout_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Different key and dtype. Will also time out.
    input_dict_4 = {
        'key': np.array(6004, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.bool],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_timeout_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: Repeated index. This is valid but will also time out.
    input_dict_5 = {
        'key': np.array(6005, dtype=np.int64),
        'indices': np.array([0, 0], dtype=np.int32),
        'dtypes': [tf.complex64],
        'capacity': 5,
        'memory_limit': 0,
        'container': 'another_container',
        'shared_name': '',
        'name': 'peek_timeout_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: Final simple case. Will time out.
    input_dict_6 = {
        'key': np.array(6006, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.int32],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_timeout_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    return list_of_inputs

generated_inputs["tf.raw_ops.MapPeek"] = tf_raw_ops_mappeek_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MapPeek' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MapPeek'.")

check_valid('tf.raw_ops.MapPeek', generated_inputs['tf.raw_ops.MapPeek'], lib="tf", suffix=0)
