
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decode_json_example_inputs():
    list_of_inputs = []

    # Input 1
    json_examples = tf.constant(['{"features": {"feature": {"f1": {"float_list": {"value": [1.0, 2.0]}}}}}'])
    name = "decode_example_1"


    list_of_inputs.append({"json_examples": json_examples, "name": name})

    # Input 2
    json_examples = tf.constant(['{"features": {"feature": {"i1": {"int64_list": {"value": [1, 2, 3]}}}}}'])
    name = "decode_example_2"


    list_of_inputs.append({"json_examples": json_examples, "name": name})

    # Input 3
    json_examples = tf.constant(['{"features": {"feature": {"s1": {"bytes_list": {"value": ["test".encode()]}}}}}'])
    name = "decode_example_3"


    list_of_inputs.append({"json_examples": json_examples, "name": name})

    # Input 4
    json_examples = tf.constant(['{"features": {"feature": {"f1": {"float_list": {"value": [1.0]}}, "i1": {"int64_list": {"value": [1]}}, "s1": {"bytes_list": {"value": ["test".encode()]}}}}}'])
    name = "decode_example_4"


    list_of_inputs.append({"json_examples": json_examples, "name": name})

    # Input 5
    json_examples = tf.constant(['{"features": {"feature": {}}}'])
    name = "decode_example_5"


    list_of_inputs.append({"json_examples": json_examples, "name": name})

    # Input 6: Multiple JSON examples
    json_examples = tf.constant(['{"features": {"feature": {"f1": {"float_list": {"value": [1.0]}}}}}', '{"features": {"feature": {"i1": {"int64_list": {"value": [1]}}}}}'])
    name = "decode_example_6"


    list_of_inputs.append({"json_examples": json_examples, "name": name})

    # Input 7: Empty string
    json_examples = tf.constant([''])
    name = "decode_example_7"


    list_of_inputs.append({"json_examples": json_examples, "name": name})

    # Input 8: Example with multiple features
    json_examples = tf.constant(['{"features": {"feature": {"f1": {"float_list": {"value": [1.0, 2.0, 3.0]}}, "i1": {"int64_list": {"value": [4, 5, 6]}}, "s1": {"bytes_list": {"value": ["a".encode(), "b".encode(), "c".encode()]}}}}}'])
    name = "decode_example_8"


    list_of_inputs.append({"json_examples": json_examples, "name": name})

    # Input 9: With FeatureList
    json_examples = tf.constant(['{"feature_lists": {"feature_list": {"f1": {"feature": [{"float_list": {"value": [1.0]}}]}}}}'])
    name = "decode_example_9"


    list_of_inputs.append({"json_examples": json_examples, "name": name})

    # Input 10: More complex FeatureList
    json_examples = tf.constant(['{"feature_lists": {"feature_list": {"f1": {"feature": [{"float_list": {"value": [1.0, 2.0]}}, {"float_list": {"value": [3.0, 4.0]}}]}}}}'])
    name = "decode_example_10"


    list_of_inputs.append({"json_examples": json_examples, "name": name})

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
