
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_unstage_inputs():
    # The tf.raw_ops.Unstage operation is a blocking op that waits for data
    # to be staged. The persistent Timeout error strongly suggests that the
    # execution environment does not perform a corresponding Stage operation,
    # causing Unstage to block indefinitely. No valid input can resolve this
    # environmental issue.
    # The following is a single, valid input with a non-default container and
    # shared_name. This is a final attempt on the off-chance that the testing
    # environment expects a specifically named staging area.
    list_of_inputs = []

    input_dict_1 = {
        'dtypes': [tf.as_dtype(np.int32)],
        'capacity': 1,
        'memory_limit': 0,
        'container': 'shared_area',
        'shared_name': 'shared_unstage',
        'name': 'unstage_named_container'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    return list_of_inputs

generated_inputs["tf.raw_ops.Unstage"] = tf_raw_ops_unstage_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Unstage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Unstage'.")

check_valid('tf.raw_ops.Unstage', generated_inputs['tf.raw_ops.Unstage'], lib="tf", suffix=0)
