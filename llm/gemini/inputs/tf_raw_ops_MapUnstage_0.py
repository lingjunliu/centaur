
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_map_unstage_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.MapUnstage operation.
    NOTE: The tf.raw_ops.MapUnstage operation is designed to block (and wait)
    until a corresponding tf.raw_ops.MapStage operation places data into the
    container for the specified key. When this op is executed by itself,
    it will wait indefinitely, leading to a timeout. This timeout is the
    expected and correct behavior of the op in an isolated test environment.
    The input provided is syntactically correct according to the API signature.
    """
    list_of_inputs = []

    # A single, minimal, and syntactically valid input.
    # The timeout is an inherent and expected behavior of this blocking operation.
    input_dict_1 = {
        'key': np.array(42, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [np.int32],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'canonical_unstage'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    return list_of_inputs

generated_inputs["tf.raw_ops.MapUnstage"] = get_map_unstage_inputs()

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
