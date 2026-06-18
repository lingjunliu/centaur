
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_strings_unicode_encode_inputs():
    list_of_inputs = []

    # Input 1: Standard UTF-8, 2D array, valid ASCII characters
    input_val = np.array([[71, 111, 111, 100], [109, 111, 114, 110]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-8",
        "errors": "replace",
        "replacement_char": 65533,
        "name": "encode_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: UTF-16-BE, 1D array
    input_val = np.array([71, 111, 111, 100], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-16-BE",
        "errors": "replace",
        "replacement_char": 65533,
        "name": "encode_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: UTF-32-BE, 3D array
    input_val = np.array([[[65, 66], [67, 68]], [[69, 70], [71, 72]]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-32-BE",
        "errors": "replace",
        "replacement_char": 65533,
        "name": "encode_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Errors "ignore" with invalid surrogate codepoint
    input_val = np.array([[65, 0xD800, 66]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-8",
        "errors": "ignore",
        "replacement_char": 65533,
        "name": "encode_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Errors "strict", with emoji characters
    input_val = np.array([[128522, 128523]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-8",
        "errors": "strict",
        "replacement_char": 65533,
        "name": "encode_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Custom replacement character for invalid codepoint
    input_val = np.array([[65, 0x110000, 66]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-8",
        "errors": "replace",
        "replacement_char": 63,
        "name": "encode_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Int32 input type, UTF-16-BE encoding
    input_val = np.array([[104, 101, 108, 108, 111]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-16-BE",
        "errors": "strict",
        "replacement_char": 65533,
        "name": "encode_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty last dimension
    input_val = np.empty((2, 0), dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-8",
        "errors": "replace",
        "replacement_char": 65533,
        "name": "encode_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Emojis with UTF-32-BE encoding
    input_val = np.array([[0x1F600, 0x1F601, 0x1F602]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-32-BE",
        "errors": "strict",
        "replacement_char": 65533,
        "name": "encode_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single character elements along the last axis
    input_val = np.array([[65], [66], [67]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-8",
        "errors": "replace",
        "replacement_char": 65533,
        "name": "encode_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strings.unicode_encode"] = tf_strings_unicode_encode_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strings.unicode_encode' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.unicode_encode'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.strings.unicode_encode', generated_inputs['tf.strings.unicode_encode'], lib="tf", suffix=0)
