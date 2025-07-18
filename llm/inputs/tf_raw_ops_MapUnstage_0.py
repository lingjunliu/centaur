
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_map_unstage_inputs():
    # The previous error was an InvalidArgumentError because the 'container'
    # string "test_container" contained an invalid character ('_').
    # Valid container names typically follow stricter naming conventions.
    # The fix is to use a simple alphanumeric string without special characters.
    # The timeout issue remains a fundamental characteristic of this op,
    # as it blocks waiting for a corresponding MapStage op. We will continue
    # to provide only one minimal input to avoid timeouts after fixing the
    # invalid argument.
    list_of_inputs = []

    input_dict = {
        'key': np.array(1, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.float32],
        'capacity': 1,
        'memory_limit': 0,
        'container': 'testcontainer',
        'shared_name': 'testsharedname',
        'name': 'TestMapUnstage'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
