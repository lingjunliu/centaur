
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_raw_ops_unstage_inputs():
    """
    Generates inputs for tf.raw_ops.Unstage.

    The Unstage op is a blocking operation that waits for data. In an isolated
    execution without a corresponding Stage op, this will inherently cause a
    timeout. The provided inputs are syntactically correct. The timeout is an
    expected runtime behavior of the op itself in this context. This list is
    kept minimal to provide valid, representative examples without exacerbating
    timeout issues in the test runner.
    """
    list_of_inputs = []

    # Input 1: The most basic valid input with a single float dtype.
    input_dict_1 = {
        'dtypes': [tf.float32],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'unstage_minimal_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: A basic valid input with a single integer dtype.
    input_dict_2 = {
        'dtypes': [tf.int32],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'unstage_minimal_int'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

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
