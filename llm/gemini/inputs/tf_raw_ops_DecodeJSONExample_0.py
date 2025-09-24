
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_raw_ops_decodejsonexample_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.DecodeJSONExample function.
    """
    list_of_inputs = []

    # These JSON strings are created from tf.train.Example protos using
    # `json_format.MessageToJson`. The format is very specific.
    # Note: bytes are Base64 encoded.
    # e.g., b'user1' -> 'dXNlcjE='
    json_1 = '{"features":{"feature":{"id":{"bytesList":{"value":["dXNlcjE="]}},"age":{"int64List":{"value":["25"]}}}}}'
    # e.g., float_list=[98.5, 99.0, 97.2]
    json_2 = '{"features":{"feature":{"scores":{"floatList":{"value":[98.5,99.0,97.2]}}}}}'
    # e.g., b'dummy_bytes' -> 'ZHVtbXlfYnl0ZXM='
    json_3 = '{"features":{"feature":{"image_data":{"bytesList":{"value":["ZHVtbXlfYnl0ZXM="]}}}}}'
    # An empty tf.train.Example()
    json_empty = '{}'
    # An example with a feature that has an empty list.
    json_empty_list = '{"features":{"feature":{"tags":{"bytesList":{}}}}}'


    # Input 1: Single JSON example in a 1D tensor
    input_dict = {
        'json_examples': np.array([json_1], dtype=object),
        'name': 'single_example_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple JSON examples in a 1D tensor
    input_dict = {
        'json_examples': np.array([json_1, json_2, json_3], dtype=object),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: A scalar (0-D) tensor
    input_dict = {
        'json_examples': np.array(json_2, dtype=object),
        'name': 'scalar_example'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: A 2D tensor of JSON examples
    input_dict = {
        'json_examples': np.array([[json_1, json_2], [json_3, json_empty]], dtype=object),
        'name': '2d_examples'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: A single, completely empty example
    input_dict = {
        'json_examples': np.array([json_empty], dtype=object),
        'name': 'empty_example'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: A tensor with a mix of regular and empty-feature examples
    input_dict = {
        'json_examples': np.array([json_1, json_empty, json_2, json_empty_list], dtype=object),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: An example with a feature that has an empty list of values
    input_dict = {
        'json_examples': np.array([json_empty_list], dtype=object),
        'name': 'empty_list_feature'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty input tensor (shape=(0,))
    input_dict = {
        'json_examples': np.array([], dtype=object),
        'name': 'empty_input_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High-dimensional tensor (3D)
    input_dict = {
        'json_examples': np.array([[[json_1], [json_2]], [[json_3], [json_empty]]], dtype=object),
        'name': '3d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: An empty 2D tensor (shape=(2,0))
    input_dict = {
        'json_examples': np.empty(shape=(2,0), dtype=object),
        'name': 'empty_2d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DecodeJSONExample"] = get_raw_ops_decodejsonexample_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DecodeJSONExample' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeJSONExample'.")

check_valid('tf.raw_ops.DecodeJSONExample', generated_inputs['tf.raw_ops.DecodeJSONExample'], lib="tf", suffix=0)
