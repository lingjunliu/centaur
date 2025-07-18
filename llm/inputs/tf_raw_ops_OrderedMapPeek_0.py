
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_raw_ops_orderedmappeek_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.OrderedMapPeek function.
    """
    list_of_inputs = []

    # Input 1: Basic case with a single dtype and default optional parameters.
    input_dict_1 = {
        'key': np.array(1, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.float32],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple dtypes.
    input_dict_2 = {
        'key': np.array(2, dtype=np.int64),
        'indices': np.array([0, 1], dtype=np.int32),
        'dtypes': [tf.int32, tf.string],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Multiple indices and multiple dtypes.
    input_dict_3 = {
        'key': np.array(3, dtype=np.int64),
        'indices': np.array([0, 1, 2], dtype=np.int32),
        'dtypes': [tf.float64, tf.int64, tf.bool],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Non-default capacity.
    input_dict_4 = {
        'key': np.array(4, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.int16],
        'capacity': 10,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Non-default memory_limit.
    input_dict_5 = {
        'key': np.array(5, dtype=np.int64),
        'indices': np.array([3], dtype=np.int32),
        'dtypes': [tf.uint8],
        'capacity': 0,
        'memory_limit': 1024,
        'container': '',
        'shared_name': '',
        'name': 'peek_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Non-default container.
    input_dict_6 = {
        'key': np.array(6, dtype=np.int64),
        'indices': np.array([0, 2], dtype=np.int32),
        'dtypes': [tf.float32, tf.complex64],
        'capacity': 0,
        'memory_limit': 0,
        'container': 'my_container',
        'shared_name': '',
        'name': 'peek_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Non-default shared_name.
    input_dict_7 = {
        'key': np.array(7, dtype=np.int64),
        'indices': np.array([1], dtype=np.int32),
        'dtypes': [tf.string],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': 'my_shared_map',
        'name': 'peek_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: All optional parameters set.
    input_dict_8 = {
        'key': np.array(8, dtype=np.int64),
        'indices': np.array([0, 1], dtype=np.int32),
        'dtypes': [tf.float32, tf.int64],
        'capacity': 50,
        'memory_limit': 2048,
        'container': 'full_container',
        'shared_name': 'full_shared_name',
        'name': 'peek_full'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Zero key value.
    input_dict_9 = {
        'key': np.array(0, dtype=np.int64),
        'indices': np.array([0, 1, 2, 3], dtype=np.int32),
        'dtypes': [tf.float32, tf.int32, tf.string, tf.bool],
        'capacity': 10,
        'memory_limit': 100,
        'container': 'container_zero',
        'shared_name': 'shared_zero',
        'name': 'peek_zero'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Empty indices tensor.
    input_dict_10 = {
        'key': np.array(10, dtype=np.int64),
        'indices': np.array([], dtype=np.int32),
        'dtypes': [tf.float32, tf.int32],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_empty_indices'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Large key value.
    input_dict_11 = {
        'key': np.array(9223372036854775807, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.bfloat16],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_large_key'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    # Input 12: Different dtypes including complex128
    input_dict_12 = {
        'key': np.array(12, dtype=np.int64),
        'indices': np.array([0, 1], dtype=np.int32),
        'dtypes': [tf.complex128, tf.uint16],
        'capacity': 12,
        'memory_limit': 12,
        'container': 'cont_12',
        'shared_name': 'shared_12',
        'name': 'peek_12'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.raw_ops.OrderedMapPeek"] = get_tf_raw_ops_orderedmappeek_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.OrderedMapPeek' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OrderedMapPeek'.")

check_valid('tf.raw_ops.OrderedMapPeek', generated_inputs['tf.raw_ops.OrderedMapPeek'], lib="tf", suffix=0)
