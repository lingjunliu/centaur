
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
from google.protobuf import json_format
import copy

def tf_io_decode_json_example_inputs():
    list_of_inputs = []

    # Helper function to create a JSON example
    def create_json_example(a_values):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "a": tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=a_values))}))
        example_json = json_format.MessageToJson(example)
        return example_json

    # Input 1: Simple example with one feature
    example_json_1 = create_json_example([1, 2, 3])
    input_dict_1 = {"json_examples": np.array(example_json_1, dtype=np.object_), "name": "example_1"}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Batch of examples
    example_json_2 = create_json_example([4, 5, 6])
    input_dict_2 = {"json_examples": np.array([example_json_1, example_json_2], dtype=np.object_), "name": "example_2"}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Example with different feature values
    example_json_3 = create_json_example([-1, 0, 1])
    input_dict_3 = {"json_examples": np.array(example_json_3, dtype=np.object_), "name": "example_3"}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Example with long feature values
    example_json_4 = create_json_example([1234567890, 9876543210])
    input_dict_4 = {"json_examples": np.array(example_json_4, dtype=np.object_), "name": "example_4"}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Example with empty feature list
    example_json_5 = create_json_example([])
    input_dict_5 = {"json_examples": np.array(example_json_5, dtype=np.object_), "name": "example_5"}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 2D Tensor of JSON strings
    example_json_6 = create_json_example([7,8,9])
    input_dict_6 = {"json_examples": np.array([[example_json_1, example_json_2], [example_json_3, example_json_6]], dtype=np.object_), "name": "example_6"}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 8: A more complex Example
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "a": tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 2, 3])),
                "b": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"test"]))
            }
        )
    )
    example_json_8 = json_format.MessageToJson(example)
    input_dict_8 = {"json_examples": np.array(example_json_8, dtype=np.object_), "name": "example_8"}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Another more complex Example with floats
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "a": tf.train.Feature(float_list=tf.train.FloatList(value=[1.0, 2.0, 3.0])),
                "b": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"test"]))
            }
        )
    )
    example_json_9 = json_format.MessageToJson(example)
    input_dict_9 = {"json_examples": np.array(example_json_9, dtype=np.object_), "name": "example_9"}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: A 3D Tensor of JSON strings
    example_json_10 = create_json_example([10,11,12])
    input_dict_10 = {"json_examples": np.array([[[example_json_1, example_json_2], [example_json_3, example_json_6]], [[example_json_10,example_json_1], [example_json_2, example_json_3]]], dtype=np.object_), "name": "example_10"}
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    input_dict_7 = {"json_examples": np.array('{"features": {}}', dtype=np.object_), "name": "example_7"}
    list_of_inputs.append(copy.deepcopy(input_dict_7))


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
