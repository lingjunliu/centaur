
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_unicode_transcode_inputs():
    list_of_inputs = []

    # Input 1: Basic case, UTF-8 to UTF-16-BE
    input_dict = {
        'input': np.array(["Hello", "TensorFlow", "2.x"], dtype=object),
        'input_encoding': "UTF-8",
        'output_encoding': "UTF-16-BE",
        'errors': "replace",
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: US-ASCII to UTF-8
    input_dict = {
        'input': np.array([b"A", b"B", b"C"], dtype=object),
        'input_encoding': "US-ASCII",
        'output_encoding': "UTF-8",
        'errors': "replace",
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: UTF-8 to UTF-32-BE with multi-byte characters
    input_dict = {
        'input': np.array(["你好", "你好"], dtype=object),
        'input_encoding': "UTF-8",
        'output_encoding': "UTF-32-BE",
        'errors': "replace",
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Error handling 'ignore' with invalid UTF-8 sequence
    input_dict = {
        'input': np.array([b"ab\x80cde", b"fgh"], dtype=object),
        'input_encoding': "UTF-8",
        'output_encoding': "UTF-16-BE",
        'errors': "ignore",
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': "test5_ignore"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Error handling 'replace' with a custom replacement character
    input_dict = {
        'input': np.array([b"invalid-\xe2\x82-char"], dtype=object), # Incomplete euro symbol
        'input_encoding': "UTF-8",
        'output_encoding': "UTF-8",
        'errors': "replace",
        'replacement_char': 63,  # '?' character
        'replace_control_characters': False,
        'name': "test6_custom_replace"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Replacing control characters
    input_dict = {
        'input': np.array([b"text\twith\ncontrol\rchars"], dtype=object),
        'input_encoding': "US-ASCII",
        'output_encoding': "UTF-8",
        'errors': "replace",
        'replacement_char': 32, # space
        'replace_control_characters': True,
        'name': "test7_replace_control"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: No-op transcode to enforce correct formatting
    input_dict = {
        'input': np.array([b"already-utf8", b"bad-\xe2-utf8"], dtype=object),
        'input_encoding': "UTF-8",
        'output_encoding': "UTF-8",
        'errors': "replace",
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': "test8_noop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D input tensor
    input_dict = {
        'input': np.array([[b"a", b"b"], [b"c", b"d"]], dtype=object),
        'input_encoding': "UTF-8",
        'output_encoding': "UTF-16-BE",
        'errors': "replace",
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': "test9_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Scalar (0-D) input tensor
    input_dict = {
        'input': np.array(b"Scalar String", dtype=object),
        'input_encoding': "UTF-8",
        'output_encoding': "UTF-32-BE",
        'errors': "replace",
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': "test10_scalar"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: BOM is consumed when endianness is not explicit
    input_dict = {
        'input': np.array([b'\xfe\xff\x00A\x00B'], dtype=object), # UTF-16BE BOM + "AB"
        'input_encoding': "UTF-16",
        'output_encoding': "UTF-8",
        'errors': "replace",
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': "test11_bom_consumed"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: BOM is preserved when endianness is explicit
    input_dict = {
        'input': np.array([b'\xfe\xff\x00A\x00B'], dtype=object), # UTF-16BE BOM + "AB"
        'input_encoding': "UTF-16-BE",
        'output_encoding': "UTF-8",
        'errors': "replace",
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': "test12_bom_preserved"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Empty input tensor
    input_dict = {
        'input': np.array([], dtype=object),
        'input_encoding': "UTF-8",
        'output_encoding': "UTF-16-BE",
        'errors': 'replace',
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': "empty_input"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.UnicodeTranscode"] = tf_raw_ops_unicode_transcode_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.UnicodeTranscode' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.UnicodeTranscode'.")

check_valid('tf.raw_ops.UnicodeTranscode', generated_inputs['tf.raw_ops.UnicodeTranscode'], lib="tf", suffix=0)
