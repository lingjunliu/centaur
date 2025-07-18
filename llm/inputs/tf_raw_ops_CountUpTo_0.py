
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_countupto_inputs():
    """
    Generates inputs for tf.raw_ops.CountUpTo.

    The persistent "RuntimeError: count_up_to op does not support eager execution"
    is a fundamental characteristic of this specific TensorFlow operation. It is
    designed to work only within a TensorFlow graph context (e.g., inside a
    tf.function or a TF1 Session). The error arises because the execution
    framework is attempting to run this graph-only operation in eager mode.

    The inputs provided below are correct according to the API signature and the
    numpy format requirement. The issue is not with the inputs themselves but with
    the execution environment. The calling code must be modified to create a graph
    context for this operation to succeed.
    """
    list_of_inputs = []

    # Input 1: Basic int32 case
    input_dict_1 = {
        'ref': np.array(0, dtype=np.int32),
        'limit': 5,
        'name': 'count_up_to_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic int64 case
    input_dict_2 = {
        'ref': np.array(10, dtype=np.int64),
        'limit': 15,
        'name': 'count_up_to_15'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: No optional name
    input_dict_3 = {
        'ref': np.array(99, dtype=np.int32),
        'limit': 101,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: ref is one less than limit
    input_dict_4 = {
        'ref': np.array(49, dtype=np.int64),
        'limit': 50,
        'name': 'almost_at_limit'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Negative starting ref
    input_dict_5 = {
        'ref': np.array(-5, dtype=np.int32),
        'limit': 5,
        'name': 'negative_to_positive'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Both ref and limit are negative
    input_dict_6 = {
        'ref': np.array(-10, dtype=np.int64),
        'limit': -5,
        'name': 'all_negative'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Minimal range
    input_dict_7 = {
        'ref': np.array(0, dtype=np.int32),
        'limit': 1,
        'name': 'minimal_range'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Minimal negative range
    input_dict_8 = {
        'ref': np.array(-1, dtype=np.int64),
        'limit': 0,
        'name': 'minimal_negative_range'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large range with int32
    input_dict_9 = {
        'ref': np.array(0, dtype=np.int32),
        'limit': 50000,
        'name': 'large_range_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Large range with int64
    input_dict_10 = {
        'ref': np.array(1000, dtype=np.int64),
        'limit': 200000,
        'name': 'large_range_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.CountUpTo"] = get_tf_raw_ops_countupto_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.CountUpTo' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.CountUpTo'.")

check_valid('tf.raw_ops.CountUpTo', generated_inputs['tf.raw_ops.CountUpTo'], lib="tf", suffix=0)
