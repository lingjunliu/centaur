
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_count_up_to_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.CountUpTo operation.
    This operation is not compatible with eager execution and requires a graph context.
    The inputs are provided in numpy format as required by the specified signature,
    even though this will lead to a runtime error in an eager execution environment.
    """
    list_of_inputs = []

    # Input 1: Basic case with int32.
    input_dict_1 = {
        'ref': np.array(0, dtype=np.int32),
        'limit': 10,
        'name': 'case_1_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case with int64.
    input_dict_2 = {
        'ref': np.array(50, dtype=np.int64),
        'limit': 60,
        'name': 'case_2_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Limit is close to ref.
    input_dict_3 = {
        'ref': np.array(9, dtype=np.int32),
        'limit': 10,
        'name': 'case_3_limit_close'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Negative ref, positive limit.
    input_dict_4 = {
        'ref': np.array(-5, dtype=np.int32),
        'limit': 5,
        'name': 'case_4_negative_ref'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Negative ref, zero limit.
    input_dict_5 = {
        'ref': np.array(-1, dtype=np.int64),
        'limit': 0,
        'name': 'case_5_zero_limit'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Both ref and limit are negative.
    input_dict_6 = {
        'ref': np.array(-20, dtype=np.int32),
        'limit': -10,
        'name': 'case_6_both_negative'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Large int64 numbers.
    input_dict_7 = {
        'ref': np.array(10000000000, dtype=np.int64),
        'limit': 10000000010,
        'name': 'case_7_large_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: No name provided.
    input_dict_8 = {
        'ref': np.array(1, dtype=np.int32),
        'limit': 2,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Wide range between ref and limit.
    input_dict_9 = {
        'ref': np.array(-1000, dtype=np.int64),
        'limit': 1000,
        'name': 'case_9_wide_range'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Large limit.
    input_dict_10 = {
        'ref': np.array(0, dtype=np.int32),
        'limit': 200000,
        'name': 'case_10_large_limit'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.CountUpTo"] = tf_raw_ops_count_up_to_inputs()

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
