
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_timestamp_inputs():
    list_of_inputs = []

    # The `FailedPreconditionError: Timestamp cannot be called when determinism is enabled`
    # is an environmental constraint, not an input validation error. The test harness
    # requires valid inputs to be generated. The following inputs are valid as per
    # the API signature, even though they will fail in a deterministic environment.

    input_dict_1 = {'name': 'timestamp_1'}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_dict_2 = {'name': 'timestamp_2'}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_dict_3 = {'name': 'time_marker_x'}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_dict_4 = {'name': 'time_marker_y'}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_dict_5 = {'name': 'start_time'}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_dict_6 = {'name': 'end_time'}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_dict_7 = {'name': 'op_timestamp'}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    input_dict_8 = {'name': 'debug_time_point'}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    input_dict_9 = {'name': 'another_unique_name'}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    input_dict_10 = {'name': 'final_unique_name'}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.Timestamp"] = tf_raw_ops_timestamp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Timestamp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Timestamp'.")

check_valid('tf.raw_ops.Timestamp', generated_inputs['tf.raw_ops.Timestamp'], lib="tf", suffix=0)
