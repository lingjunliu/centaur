
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_map_unstage_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.MapUnstage operation.
    NOTE: This operation is inherently blocking and WILL cause a timeout or an
    error if the corresponding tf.raw_ops.MapStage operation has not been
    called beforehand. The provided inputs are syntactically valid to avoid the
    InvalidArgumentError related to container names. Container names must not
    contain characters like underscores.
    """
    list_of_inputs = []

    # Input 1: Basic case with default empty container and shared_name.
    input_dict_1 = {
        'key': np.array(1, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.float32],
        'capacity': 1,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': "Unstage_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Different key and multiple dtypes, also with default empty container.
    input_dict_2 = {
        'key': np.array(2, dtype=np.int64),
        'indices': np.array([0, 1], dtype=np.int32),
        'dtypes': [tf.int64, tf.bool],
        'capacity': 2,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': "Unstage_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Using a valid, non-empty alphanumeric container name.
    input_dict_3 = {
        'key': np.array(3, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.string],
        'capacity': 1,
        'memory_limit': 1024,
        'container': 'validcontainer',
        'shared_name': 'validsharedname',
        'name': "Unstage_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.MapUnstage"] = tf_raw_ops_map_unstage_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MapUnstage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MapUnstage'.")

check_valid('tf.raw_ops.MapUnstage', generated_inputs['tf.raw_ops.MapUnstage'], lib="tf", suffix=0)
