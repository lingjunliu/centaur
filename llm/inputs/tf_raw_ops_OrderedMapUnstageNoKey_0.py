
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_orderedmapunstagenokey_inputs():
    list_of_inputs = []

    # Input 1: Basic case with a single float type.
    # The op is inherently blocking and will time out in isolated execution.
    # These inputs are syntactically valid for graph construction.
    input_dict_1 = {
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.float32],
        'capacity': 2,
        'memory_limit': 0,
        'container': '',
        'shared_name': 'map_name_1',
        'name': 'test_unstage_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple dtypes.
    input_dict_2 = {
        'indices': np.array([0, 1], dtype=np.int32),
        'dtypes': [tf.int64, tf.string],
        'capacity': 5,
        'memory_limit': 0,
        'container': '',
        'shared_name': 'map_name_2',
        'name': 'test_unstage_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Using a private container name.
    input_dict_3 = {
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.bool],
        'capacity': 1,
        'memory_limit': 0,
        'container': 'private_map_3',
        'shared_name': '',
        'name': 'test_unstage_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Unbounded capacity and memory.
    input_dict_4 = {
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.complex64],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': 'map_name_4',
        'name': 'test_unstage_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Non-zero memory limit.
    input_dict_5 = {
        'indices': np.array([0, 1], dtype=np.int32),
        'dtypes': [tf.int16, tf.uint8],
        'capacity': 10,
        'memory_limit': 4096,
        'container': 'private_map_5',
        'shared_name': '',
        'name': 'test_unstage_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["tf.raw_ops.OrderedMapUnstageNoKey"] = tf_raw_ops_orderedmapunstagenokey_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.OrderedMapUnstageNoKey' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OrderedMapUnstageNoKey'.")

check_valid('tf.raw_ops.OrderedMapUnstageNoKey', generated_inputs['tf.raw_ops.OrderedMapUnstageNoKey'], lib="tf", suffix=0)
