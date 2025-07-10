
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
from google.protobuf import json_format
import numpy as np

def tf_io_decode_json_example_inputs():
    list_of_inputs = []

    example1 = tf.train.Example(features=tf.train.Features(feature={"a": tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 1, 3]))}))
    example_json1 = json_format.MessageToJson(example1)

    example2 = tf.train.Example(features=tf.train.Features(feature={"b": tf.train.Feature(float_list=tf.train.FloatList(value=[1.0, 2.0, 3.0]))}))
    example_json2 = json_format.MessageToJson(example2)

    example3 = tf.train.Example(features=tf.train.Features(feature={"c": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'hello', b'world']))}))
    example_json3 = json_format.MessageToJson(example3)

    example4 = tf.train.Example(features=tf.train.Features(feature={"d": tf.train.Feature(int64_list=tf.train.Int64List(value=[-1, -2, -3]))}))
    example_json4 = json_format.MessageToJson(example4)
    
    example5 = tf.train.Example(features=tf.train.Features(feature={"e": tf.train.Feature(float_list=tf.train.FloatList(value=[-1.0, -2.0, -3.0]))}))
    example_json5 = json_format.MessageToJson(example5)

    example6 = tf.train.Example(features=tf.train.Features(feature={"f": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'']))}))
    example_json6 = json_format.MessageToJson(example6)

    example7 = tf.train.Example(features=tf.train.Features(feature={"g": tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))}))
    example_json7 = json_format.MessageToJson(example7)
    
    example8 = tf.train.Example(features=tf.train.Features(feature={"h": tf.train.Feature(float_list=tf.train.FloatList(value=[1.5]))}))
    example_json8 = json_format.MessageToJson(example8)

    example9 = tf.train.Example(features=tf.train.Features(feature={"i": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'test']))}))
    example_json9 = json_format.MessageToJson(example9)
    
    example10 = tf.train.Example(features=tf.train.Features(feature={"j": tf.train.Feature(int64_list=tf.train.Int64List(value=[1,2,3,4,5]))}))
    example_json10 = json_format.MessageToJson(example10)

    # Input 1
    input_dict = {"json_examples": np.array(example_json1, dtype=np.string_), "name": np.array("example1", dtype=np.string_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {"json_examples": np.array([example_json1, example_json2], dtype=np.string_), "name": np.array("example2", dtype=np.string_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {"json_examples": np.array([[example_json1, example_json2], [example_json3, example_json4]], dtype=np.string_), "name": np.array("example3", dtype=np.string_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {"json_examples": np.array(example_json4, dtype=np.string_), "name": np.array("", dtype=np.string_)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {"json_examples": np.array(example_json5, dtype=np.string_), "name": np.array("example5", dtype=np.string_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {"json_examples": np.array(example_json6, dtype=np.string_), "name": np.array("example6", dtype=np.string_)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {"json_examples": np.array(example_json7, dtype=np.string_), "name": np.array("example7", dtype=np.string_)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {"json_examples": np.array(example_json8, dtype=np.string_), "name": np.array("example8", dtype=np.string_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {"json_examples": np.array(example_json9, dtype=np.string_), "name": np.array("example9", dtype=np.string_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {"json_examples": np.array(example_json10, dtype=np.string_), "name": np.array("example10", dtype=np.string_)}
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
