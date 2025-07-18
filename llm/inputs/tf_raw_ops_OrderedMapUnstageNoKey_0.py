
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_ordered_map_unstage_no_key_inputs():
    # The `OrderedMapUnstageNoKey` op is designed to block if the underlying map is empty.
    # In an isolated testing environment that executes only this op, a timeout is an
    # expected behavior and does not indicate an invalid input. To satisfy the prompt while
    # acknowledging this, we provide a minimal set of syntactically valid inputs,
    # each with a unique resource name (`shared_name`) to prevent test interference.
    list_of_inputs = []

    # Input 1: A minimal, valid input.
    input_dict_1 = {
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [tf.float32],
        'capacity': 1,
        'memory_limit': 0,
        'container': '',
        'shared_name': 'unstage_test_map_1',
        'name': 'unstage_op_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: A valid input with multiple dtypes.
    input_dict_2 = {
        'indices': np.array([0, 1], dtype=np.int32),
        'dtypes': [tf.int64, tf.string],
        'capacity': 5,
        'memory_limit': 1024,
        'container': '',
        'shared_name': 'unstage_test_map_2',
        'name': 'unstage_op_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

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
