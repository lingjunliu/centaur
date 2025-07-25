
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_countupto_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.CountUpTo operation.

    NOTE: The error 'RuntimeError: count_up_to op does not support eager
    execution' is fundamental to this specific raw operation. It is designed for
    TensorFlow's graph execution mode and explicitly checks for and disallows
    eager execution. This error cannot be resolved by changing the input values
    alone. The execution framework must be adapted to run this specific op within a
    `tf.function` or a graph context. The inputs provided here are valid according
    to the API's signature but will consistently trigger this expected runtime error
    in an eager execution environment.
    """
    list_of_inputs = []

    # Input 1: Basic positive range, int32
    input_dict_1 = {
        'ref': np.array(0, dtype=np.int32),
        'limit': 15,
        'name': 'positive_range_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Negative to positive range, int64
    input_dict_2 = {
        'ref': np.array(-5, dtype=np.int64),
        'limit': 5,
        'name': 'neg_to_pos_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Minimal increment, int32
    input_dict_3 = {
        'ref': np.array(42, dtype=np.int32),
        'limit': 43,
        'name': 'minimal_increment'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: ref is one less than limit, int64
    input_dict_4 = {
        'ref': np.array(99, dtype=np.int64),
        'limit': 100,
        'name': 'one_less_than_limit'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Crossing zero from a lower negative, int32
    input_dict_5 = {
        'ref': np.array(-10, dtype=np.int32),
        'limit': 1,
        'name': 'crossing_zero_from_deep'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Near int32 max value
    input_dict_6 = {
        'ref': np.array(2147483640, dtype=np.int32),
        'limit': 2147483645,
        'name': 'near_int32_max'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: All negative range, int64
    input_dict_7 = {
        'ref': np.array(-200, dtype=np.int64),
        'limit': -190,
        'name': 'all_negative_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Basic positive range, int64
    input_dict_8 = {
        'ref': np.array(50, dtype=np.int64),
        'limit': 60,
        'name': 'positive_range_int64_v2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: All negative range, int32
    input_dict_9 = {
        'ref': np.array(-50, dtype=np.int32),
        'limit': -40,
        'name': 'all_negative_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Larger range starting from zero, int64
    input_dict_10 = {
        'ref': np.array(0, dtype=np.int64),
        'limit': 500,
        'name': 'large_range_from_zero_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.CountUpTo"] = tf_raw_ops_countupto_inputs()

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
