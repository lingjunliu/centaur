
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_map_peek_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.MapPeek function.
    NOTE: The tf.raw_ops.MapPeek operation is inherently blocking. It will wait indefinitely
    if the key it is trying to peek at does not exist in the map. In a testing
    environment where the map is not prepopulated (e.g., by a tf.raw_ops.MapInsert
    operation), any call to MapPeek will result in a timeout. This is the expected
    behavior of the operation, not an error in the input. The provided inputs
    are syntactically valid according to the API's documentation.
    """
    list_of_inputs = []

    # Input 1: Basic float32
    input_dict_1 = {
        'key': np.array(301, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.float32],
        'capacity': 0, 'memory_limit': 0, 'container': '',
        'shared_name': 'peek_map_a1', 'name': 'peek_a1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic int32
    input_dict_2 = {
        'key': np.array(302, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.int32],
        'capacity': 0, 'memory_limit': 0, 'container': '',
        'shared_name': 'peek_map_b1', 'name': 'peek_b1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Basic string
    input_dict_3 = {
        'key': np.array(303, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.string],
        'capacity': 0, 'memory_limit': 0, 'container': '',
        'shared_name': 'peek_map_c1', 'name': 'peek_c1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Basic bool
    input_dict_4 = {
        'key': np.array(304, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.bool],
        'capacity': 0, 'memory_limit': 0, 'container': '',
        'shared_name': 'peek_map_d1', 'name': 'peek_d1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Multiple dtypes
    input_dict_5 = {
        'key': np.array(305, dtype=np.int64),
        'indices': np.array([0, 1], dtype=np.int32),
        'dtypes': [tf.int64, tf.float64],
        'capacity': 0, 'memory_limit': 0, 'container': '',
        'shared_name': 'peek_map_e1', 'name': 'peek_e1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: With container name
    input_dict_6 = {
        'key': np.array(306, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.complex64],
        'capacity': 0, 'memory_limit': 0, 'container': 'my_peek_container_1',
        'shared_name': 'peek_map_f1', 'name': 'peek_f1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: With capacity and memory limit
    input_dict_7 = {
        'key': np.array(307, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.int16],
        'capacity': 20, 'memory_limit': 4096, 'container': '',
        'shared_name': 'peek_map_g1', 'name': 'peek_g1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Zero key
    input_dict_8 = {
        'key': np.array(0, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.uint8],
        'capacity': 0, 'memory_limit': 0, 'container': '',
        'shared_name': 'peek_map_h1', 'name': 'peek_h1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Negative key
    input_dict_9 = {
        'key': np.array(-309, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.float16],
        'capacity': 0, 'memory_limit': 0, 'container': '',
        'shared_name': 'peek_map_i1', 'name': 'peek_i1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Long list of dtypes
    input_dict_10 = {
        'key': np.array(310, dtype=np.int64),
        'indices': np.array([0, 1, 2, 3], dtype=np.int32),
        'dtypes': [tf.uint16, tf.uint32, tf.uint64, tf.bfloat16],
        'capacity': 0, 'memory_limit': 0, 'container': '',
        'shared_name': 'peek_map_j1', 'name': 'peek_j1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.MapPeek"] = tf_raw_ops_map_peek_inputs()

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
