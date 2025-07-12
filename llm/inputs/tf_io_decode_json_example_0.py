
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
from google.protobuf import json_format
import copy

def tf_io_decode_json_example_inputs():
    list_of_inputs = []

    # Helper function to create example JSON
    def create_example_json(values):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "a": tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=values))}))
        return json_format.MessageToJson(example)

    # Input 1: Simple valid JSON
    example_json = create_example_json([1, 2, 3])
    input_dict = {"json_examples": tf.constant(example_json, dtype=tf.string), "name": "example1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of valid JSONs
    example_json_batch = [create_example_json([4, 5, 6]), create_example_json([7, 8, 9])]
    input_dict = {"json_examples": tf.constant(example_json_batch, dtype=tf.string), "name": "example2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional array of valid JSONs
    example_json_array = np.array([[create_example_json([10, 11, 12]), create_example_json([13, 14, 15])],
                                   [create_example_json([16, 17, 18]), create_example_json([19, 20, 21])]])
    input_dict = {"json_examples": tf.constant(example_json_array.tolist(), dtype=tf.string), "name": "example3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: JSON with different feature
    example_json_different_feature = json_format.MessageToJson(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "b": tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=[1.0, 2.0, 3.0]))})))
    input_dict = {"json_examples": tf.constant(example_json_different_feature, dtype=tf.string), "name": "example4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty list
    example_json_empty_list = create_example_json([])
    input_dict = {"json_examples": tf.constant(example_json_empty_list, dtype=tf.string), "name": "example5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger values
    example_json_large_values = create_example_json([2**31 - 1, 2**31 - 2, 2**31 - 3])
    input_dict = {"json_examples": tf.constant(example_json_large_values, dtype=tf.string), "name": "example6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single Value
    example_json_single_value = create_example_json([100])
    input_dict = {"json_examples": tf.constant(example_json_single_value, dtype=tf.string), "name": "example7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multiple features
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "a": tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 2, 3])),
                "b": tf.train.Feature(float_list=tf.train.FloatList(value=[4.0, 5.0, 6.0]))
            }))

    example_json_multiple_features = json_format.MessageToJson(example)
    input_dict = {"json_examples": tf.constant(example_json_multiple_features, dtype=tf.string), "name": "example8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty Feature
    example = tf.train.Example(features=tf.train.Features(feature={}))
    example_json_empty_feature = json_format.MessageToJson(example)

    input_dict = {"json_examples": tf.constant(example_json_empty_feature, dtype=tf.string), "name": "example9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10: Empty JSON string
    input_dict = {"json_examples": tf.constant("", dtype=tf.string), "name": "example10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_json_example"] = tf_io_decode_json_example_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.decode_json_example' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_json_example'.")

check_valid('tf.io.decode_json_example', generated_inputs['tf.io.decode_json_example'], lib="tf", suffix=0)
