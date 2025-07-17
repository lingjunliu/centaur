
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MergeSummary_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input with a single summary
    inputs = [b'\n\x06value_1\x12\x04data']
    input_dict = {"inputs": inputs, "name": "summary_merge_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two summaries to merge
    inputs = [
        b'\n\x06value_1\x12\x04data',
        b'\n\x06value_2\x12\x04data2'
    ]
    input_dict = {"inputs": inputs, "name": "summary_merge_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty summaries
    inputs = [b'', b'']
    input_dict = {"inputs": inputs, "name": "summary_merge_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Three summaries to merge
    inputs = [
        b'\n\x06value_1\x12\x04data',
        b'\n\x06value_2\x12\x04data2',
        b'\n\x06value_3\x12\x04data3'
    ]
    input_dict = {"inputs": inputs, "name": "summary_merge_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Longer summary strings
    inputs = [
        b'\n\x06value_1\x12\x04data' * 10,
        b'\n\x06value_2\x12\x04data2' * 10
    ]
    input_dict = {"inputs": inputs, "name": "summary_merge_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single input with longer summary
    inputs = [b'\n\x06value_1\x12\x04data' * 20]
    input_dict = {"inputs": inputs, "name": "summary_merge_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Unicode summary strings (encoded as bytes)
    inputs = [b'\n\x06value_1\x12\x04\xe4\xbd\xa0\xe5\xa5\xbd']
    input_dict = {"inputs": inputs, "name": "summary_merge_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: More unicode
    inputs = [
        b'\n\x06value_1\x12\x04\xe4\xbd\xa0\xe5\xa5\xbd',
        b'\n\x06value_2\x12\x04\xe6\xac\xa2\xe8\xbf\x8e',
    ]
    input_dict = {"inputs": inputs, "name": "summary_merge_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  Different valid summary structures
    inputs = [
        b'\n\x12\x0bvalue_name\x12\x03tag\x1a\x04data',
        b'\n\x12\x0bvalue_name2\x12\x03tag2\x1a\x04data2'
    ]
    input_dict = {"inputs": inputs, "name": "summary_merge_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty name
    inputs = [b'\n\x06value_1\x12\x04data']
    input_dict = {"inputs": inputs, "name": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MergeSummary"] = tf_raw_ops_MergeSummary_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MergeSummary' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MergeSummary'.")

check_valid('tf.raw_ops.MergeSummary', generated_inputs['tf.raw_ops.MergeSummary'], lib="tf", suffix=0)
