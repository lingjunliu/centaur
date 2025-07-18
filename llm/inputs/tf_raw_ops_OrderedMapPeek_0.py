
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_orderedmappeek_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.OrderedMapPeek operation.
    NOTE: The timeout error is an expected consequence of this operation's blocking
    nature when run in isolation without a corresponding staging op. The 'Indices are empty'
    error from the previous attempt is fixed by ensuring the 'indices' tensor is never empty.
    The provided inputs are syntactically valid but are expected to time out.
    """
    list_of_inputs = []

    # Input 1: Basic case. Peeks at the first component (index 0) of a two-component value.
    input_dict_1 = {
        'key': np.array(1, dtype=np.int64),
        'indices': np.array(0, dtype=np.int32),
        'dtypes': [tf.float32, tf.int32],
        'capacity': 10,
        'memory_limit': 0,
        'container': '',
        'shared_name': 'map_1',
        'name': 'peek_basic'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Peeking at multiple components from a multi-component value using a vector for indices.
    input_dict_2 = {
        'key': np.array(2, dtype=np.int64),
        'indices': np.array([2, 0], dtype=np.int32),
        'dtypes': [tf.string, tf.bool, tf.int64],
        'capacity': 10,
        'memory_limit': 0,
        'container': '',
        'shared_name': 'map_2',
        'name': 'peek_multiple_components'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Using a container and specifying memory limit.
    input_dict_3 = {
        'key': np.array(3, dtype=np.int64),
        'indices': np.array(0, dtype=np.int32),
        'dtypes': [tf.complex128],
        'capacity': 5,
        'memory_limit': 2048,
        'container': 'my_container_3',
        'shared_name': 'map_3',
        'name': 'peek_with_container_and_mem'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Peeking at all components in reverse order.
    input_dict_4 = {
        'key': np.array(4, dtype=np.int64),
        'indices': np.array([3, 2, 1, 0], dtype=np.int32),
        'dtypes': [tf.float32, tf.int32, tf.string, tf.bool],
        'capacity': 20,
        'memory_limit': 0,
        'container': '',
        'shared_name': 'map_4',
        'name': 'peek_all_reversed'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: High capacity and a single component value.
    input_dict_5 = {
        'key': np.array(5, dtype=np.int64),
        'indices': np.array(0, dtype=np.int32),
        'dtypes': [tf.uint8],
        'capacity': 100,
        'memory_limit': 0,
        'container': '',
        'shared_name': 'map_5',
        'name': 'peek_high_capacity'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["tf.raw_ops.OrderedMapPeek"] = tf_raw_ops_orderedmappeek_inputs()

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
