
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ordered_map_unstage_no_key_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.OrderedMapUnstageNoKey function.
    """
    list_of_inputs = []

    # Input 1: Basic case with a single float type
    input_1 = {
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.float32],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'basic_unstage'
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    # Input 2: Multiple dtypes
    input_2 = {
        'indices': np.array([0, 1], dtype=np.int32),
        'dtypes': [tf.float64, tf.int32],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'multi_dtype_unstage'
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    # Input 3: Non-zero capacity and more dtypes
    input_3 = {
        'indices': np.array([0, 1, 2], dtype=np.int32),
        'dtypes': [tf.string, tf.bool, tf.int8],
        'capacity': 100,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'unstage_with_capacity'
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    # Input 4: Non-zero memory limit and complex type
    input_4 = {
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.complex64],
        'capacity': 0,
        'memory_limit': 1024 * 1024, # 1MB
        'container': '',
        'shared_name': '',
        'name': 'unstage_with_mem_limit'
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    # Input 5: Non-empty container
    input_5 = {
        'indices': np.array([0, 1], dtype=np.int32),
        'dtypes': [tf.uint8, tf.int16],
        'capacity': 10,
        'memory_limit': 0,
        'container': 'my_map_container',
        'shared_name': '',
        'name': 'unstage_in_container'
    }
    list_of_inputs.append(copy.deepcopy(input_5))

    # Input 6: Non-empty shared_name (and container)
    input_6 = {
        'indices': np.array([0, 1, 2, 3], dtype=np.int32),
        'dtypes': [tf.float32, tf.float32, tf.int64, tf.int64],
        'capacity': 50,
        'memory_limit': 0,
        'container': 'shared_container_for_map',
        'shared_name': 'shared_map_name',
        'name': 'shared_unstage'
    }
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: Special float type bfloat16
    input_7 = {
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.bfloat16],
        'capacity': 5,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'bfloat16_unstage'
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    # Input 8: All optional arguments set with complex types
    input_8 = {
        'indices': np.array([0, 1, 2], dtype=np.int32),
        'dtypes': [tf.complex128, tf.float16, tf.uint32],
        'capacity': 200,
        'memory_limit': 2 * 1024 * 1024, # 2MB
        'container': 'fully_specified_container',
        'shared_name': 'fully_specified_name',
        'name': 'complex_full_unstage'
    }
    list_of_inputs.append(copy.deepcopy(input_8))

    # Input 9: Long list of dtypes
    input_9 = {
        'indices': np.array([0, 1, 2, 3, 4], dtype=np.int32),
        'dtypes': [tf.int64, tf.int32, tf.int16, tf.int8, tf.uint8],
        'capacity': 10,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'long_list_unstage'
    }
    list_of_inputs.append(copy.deepcopy(input_9))

    # Input 10: Minimal capacity and memory_limit
    input_10 = {
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.uint64],
        'capacity': 1,
        'memory_limit': 1,
        'container': '',
        'shared_name': '',
        'name': 'minimal_resource_unstage'
    }
    list_of_inputs.append(copy.deepcopy(input_10))

    # Input 11: Quantized types
    input_11 = {
        'indices': np.array([0, 1], dtype=np.int32),
        'dtypes': [tf.qint8, tf.quint8],
        'capacity': 0,
        'memory_limit': 0,
        'container': 'quant_container',
        'shared_name': 'quant_map',
        'name': 'quantized_unstage'
    }
    list_of_inputs.append(copy.deepcopy(input_11))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.OrderedMapUnstageNoKey"] = tf_raw_ops_ordered_map_unstage_no_key_inputs()

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
