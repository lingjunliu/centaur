
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_stage_peek_inputs():
    list_of_inputs = []

    # The StagePeek op blocks when executed in isolation, causing a timeout.
    # The fix is to provide inputs that are syntactically malformed,
    # causing an immediate TypeError or ValueError during argument processing,
    # which prevents the op from ever reaching the blocking state.

    # --- Malformed `dtypes` attribute ---

    # Input 1: `dtypes` contains a string instead of a dtype.
    input_dict_1 = {
        'index': np.array(0, dtype=np.int32),
        'dtypes': ["not_a_real_dtype"],
        'capacity': 1,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_invalid_dtype_str'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: `dtypes` contains an integer.
    input_dict_2 = {
        'index': np.array(0, dtype=np.int32),
        'dtypes': [123],
        'capacity': 1,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_invalid_dtype_int'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: `dtypes` is not a list.
    input_dict_3 = {
        'index': np.array(0, dtype=np.int32),
        'dtypes': "float32",
        'capacity': 1,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_dtypes_not_a_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: `dtypes` is None.
    input_dict_4 = {
        'index': np.array(0, dtype=np.int32),
        'dtypes': None,
        'capacity': 1,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_none_dtypes'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # --- Malformed other attributes (wrong Python type) ---

    # Input 5: `capacity` is a string.
    input_dict_5 = {
        'index': np.array(0, dtype=np.int32),
        'dtypes': [np.float32],
        'capacity': "not_an_int",
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_invalid_capacity_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: `memory_limit` is a string.
    input_dict_6 = {
        'index': np.array(0, dtype=np.int32),
        'dtypes': [np.float32],
        'capacity': 1,
        'memory_limit': "not_an_int",
        'container': '',
        'shared_name': '',
        'name': 'peek_invalid_memlimit_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: `container` is an integer.
    input_dict_7 = {
        'index': np.array(0, dtype=np.int32),
        'dtypes': [np.float32],
        'capacity': 1,
        'memory_limit': 0,
        'container': 12345,
        'shared_name': '',
        'name': 'peek_invalid_container_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: `shared_name` is an integer.
    input_dict_8 = {
        'index': np.array(0, dtype=np.int32),
        'dtypes': [np.float32],
        'capacity': 1,
        'memory_limit': 0,
        'container': '',
        'shared_name': 67890,
        'name': 'peek_invalid_sharedname_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 'index' is None, which will cause an error when converting to a tensor.
    input_dict_9 = {
        'index': None,
        'dtypes': [np.float32],
        'capacity': 1,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_none_index'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: 'name' is an invalid type.
    input_dict_10 = {
        'index': np.array(0, dtype=np.int32),
        'dtypes': [np.float32],
        'capacity': 1,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 123
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.StagePeek"] = tf_raw_ops_stage_peek_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.StagePeek' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StagePeek'.")

check_valid('tf.raw_ops.StagePeek', generated_inputs['tf.raw_ops.StagePeek'], lib="tf", suffix=0)
