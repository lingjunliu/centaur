
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_strings_unicode_decode_with_offsets_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D UTF-8 input
    input_dict = {
        'input': np.array([b"hello", b"world"], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'replace',
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': 'basic_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Docstring example with multi-byte characters
    input_dict = {
        'input': np.array([s.encode('utf8') for s in ('G\xf6\xf6dnight', '\U0001f60a')], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'replace',
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': 'docstring_example'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D UTF-8 input with 'strict' errors
    input_dict = {
        'input': np.array([[b"alpha", b"beta"], [b"gamma", b"delta"]], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'strict',
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': '2d_strict'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Invalid UTF-8 sequence with 'replace' and custom char
    input_dict = {
        'input': np.array([b"start\xfaend"], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'replace',
        'replacement_char': ord('?'),
        'replace_control_characters': False,
        'name': 'error_replace_custom'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Invalid UTF-8 sequence with 'ignore'
    input_dict = {
        'input': np.array([b"begin\xfaworld\xfe"], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'ignore',
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': 'error_ignore'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different encoding (UTF-16-BE)
    input_dict = {
        'input': np.array([s.encode('utf-16-be') for s in ('Tensor', 'Flow')], dtype=object),
        'input_encoding': 'UTF-16-BE',
        'errors': 'strict',
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': 'utf16be_encoding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Replacing control characters
    input_dict = {
        'input': np.array([b'text\x01with\x0fcontrol\x1fchars'], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'replace',
        'replacement_char': 35,
        'replace_control_characters': True,
        'name': 'replace_control_chars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar (0-D) input
    input_dict = {
        'input': np.array(b'Scalar\xe2\x82\xacInput', dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'replace',
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': 'scalar_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Input with empty strings
    input_dict = {
        'input': np.array([b"first", b"", b"third"], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'strict',
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': 'with_empty_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed valid and invalid sequences
    input_dict = {
        'input': np.array([b'valid', b'\xc3\x28', b'another_valid'], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'replace',
        'replacement_char': 0,
        'replace_control_characters': False,
        'name': 'mixed_validity'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Different encoding (UTF-32-LE)
    input_dict = {
        'input': np.array([s.encode('utf-32-le') for s in ('python',)], dtype=object),
        'input_encoding': 'UTF-32-LE',
        'errors': 'strict',
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': 'utf32le_encoding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Empty input tensor
    input_dict = {
        'input': np.array([], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'strict',
        'replacement_char': 65533,
        'replace_control_characters': False,
        'name': 'empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strings.unicode_decode_with_offsets"] = tf_strings_unicode_decode_with_offsets_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strings.unicode_decode_with_offsets' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.unicode_decode_with_offsets'.")

check_valid('tf.strings.unicode_decode_with_offsets', generated_inputs['tf.strings.unicode_decode_with_offsets'], lib="tf", suffix=0)
