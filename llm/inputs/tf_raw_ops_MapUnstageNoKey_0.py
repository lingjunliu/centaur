
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_map_unstage_no_key_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.MapUnstageNoKey operation.
    NOTE: This operation is inherently blocking and will cause a timeout if the
    container is empty. The provided inputs are syntactically valid. The previous
    error was due to invalid characters (underscores) in the 'container' and
    'shared_name' arguments. These have been replaced with valid alphanumeric strings.
    """
    list_of_inputs = []

    # Input 1: Basic case with a valid container and shared name.
    input_dict_1 = {
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.float32],
        'capacity': 10,
        'memory_limit': 1024,
        'container': 'containerone',
        'shared_name': 'sharednameone',
        'name': 'unstage_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple dtypes with a valid container name.
    input_dict_2 = {
        'indices': np.array([0, 1], dtype=np.int32),
        'dtypes': [tf.int32, tf.bool],
        'capacity': 20,
        'memory_limit': 2048,
        'container': 'containertwo',
        'shared_name': '',
        'name': 'unstage_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Using a different set of indices and valid names.
    input_dict_3 = {
        'indices': np.array([5, 6, 7], dtype=np.int32),
        'dtypes': [tf.float64, tf.int8, tf.int16],
        'capacity': 5,
        'memory_limit': 4096,
        'container': 'containerthree',
        'shared_name': 'sharednamethree',
        'name': 'unstage_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Default container/shared_name (empty string)
    input_dict_4 = {
        'indices': np.array([10], dtype=np.int32),
        'dtypes': [tf.int32],
        'capacity': 500,
        'memory_limit': 1,
        'container': '',
        'shared_name': '',
        'name': 'unstage_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: More components to unstage with a valid container name.
    input_dict_5 = {
        'indices': np.array([0, 2, 4, 6], dtype=np.int32),
        'dtypes': [tf.float32, tf.int32, tf.float32, tf.bool],
        'capacity': 100,
        'memory_limit': 8192,
        'container': 'containerfive',
        'shared_name': '',
        'name': 'unstage_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Minimal positive capacity with a valid shared name.
    input_dict_6 = {
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.bool],
        'capacity': 1,
        'memory_limit': 256,
        'container': '',
        'shared_name': 'sharednamesix',
        'name': 'unstage_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Scalar index
    input_dict_7 = {
        'indices': np.array(0, dtype=np.int32),
        'dtypes': [tf.int64],
        'capacity': 2,
        'memory_limit': 512,
        'container': 'containerseven',
        'shared_name': '',
        'name': 'unstage_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs["tf.raw_ops.MapUnstageNoKey"] = tf_raw_ops_map_unstage_no_key_inputs()

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
