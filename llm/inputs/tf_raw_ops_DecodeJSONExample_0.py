
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decode_json_example_inputs():
    list_of_inputs = []

    # Input 1: Simple valid JSON example
    json_examples = np.array([
        '{"features": {"feature": {"f1": {"float_list": {"value": [1.0, 2.0]}}}}}'
    ], dtype=np.dtype('U'))
    input_dict = {"json_examples": json_examples, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple JSON examples
    json_examples = np.array([
        '{"features": {"feature": {"f1": {"float_list": {"value": [1.0]}}}}}',
        '{"features": {"feature": {"f2": {"int64_list": {"value": [2]}}}}}'
    ], dtype=np.dtype('U'))
    input_dict = {"json_examples": json_examples, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Example with int64_list
    json_examples = np.array([
        '{"features": {"feature": {"f1": {"int64_list": {"value": [1, 2, 3]}}}}}'
    ], dtype=np.dtype('U'))
    input_dict = {"json_examples": json_examples, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Example with bytes_list
    json_examples = np.array([
        '{"features": {"feature": {"f1": {"bytes_list": {"value": ["abc", "def"]}}}}}'
    ], dtype=np.dtype('U'))
    input_dict = {"json_examples": json_examples, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Example with multiple features
    json_examples = np.array([
        '{"features": {"feature": {"f1": {"float_list": {"value": [1.0]}}, "f2": {"int64_list": {"value": [2]}}}}}'
    ], dtype=np.dtype('U'))
    input_dict = {"json_examples": json_examples, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Example with string value in bytes_list
    json_examples = np.array([
        '{"features": {"feature": {"f1": {"bytes_list": {"value": ["test_string"]}}}}}'
    ], dtype=np.dtype('U'))
    input_dict = {"json_examples": json_examples, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Example with name
    json_examples = np.array([
        '{"features": {"feature": {"f1": {"float_list": {"value": [1.0]}}}}}'
    ], dtype=np.dtype('U'))
    input_dict = {"json_examples": json_examples, "name": "my_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty feature list
    json_examples = np.array(['{"features": {"feature": {}}}'], dtype=np.dtype('U'))
    input_dict = {"json_examples": json_examples, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More complex example
    json_examples = np.array([
        '{"features": {"feature": {"f1": {"float_list": {"value": [1.0, 2.0, 3.0]}}, "f2": {"int64_list": {"value": [4, 5]}}, "f3": {"bytes_list": {"value": ["abc", "def", "ghi"]}}}}}'
    ], dtype=np.dtype('U'))
    input_dict = {"json_examples": json_examples, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another example
    json_examples = np.array([
        '{"features": {"feature": {"feature1": {"float_list": {"value": [3.14]}}, "feature2": {"int64_list": {"value": [42]}}}}}'
    ], dtype=np.dtype('U'))
    input_dict = {"json_examples": json_examples, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DecodeJSONExample"] = tf_raw_ops_decode_json_example_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodeJSONExample' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeJSONExample'.")

check_valid('tf.raw_ops.DecodeJSONExample', generated_inputs['tf.raw_ops.DecodeJSONExample'], lib="tf", suffix=0)
