
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_strings_unicode_split_inputs():
    list_of_inputs = []

    # Input 1: Basic UTF-8 strings
    input_dict = {
        'input': np.array([b"hello", b"world", b"tensorflow"], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'strict',
        'replacement_char': 65533,
        'name': 'basic_utf8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multi-byte UTF-8 characters (German and Emoji)
    input_dict = {
        'input': np.array([s.encode('utf8') for s in ('G\xf6\xf6dnight', '\U0001f60a')], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'strict',
        'replacement_char': 65533,
        'name': 'multibyte_utf8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar input
    input_dict = {
        'input': np.array(b'scalar_string', dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'strict',
        'replacement_char': 65533,
        'name': 'scalar_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of strings
    input_dict = {
        'input': np.array([[b"a", b"b"], [b"c", b"d"]], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'strict',
        'replacement_char': 65533,
        'name': '2d_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Error handling 'replace' with an invalid UTF-8 sequence
    input_dict = {
        'input': np.array([b'valid', b'in\xffvalid'], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'replace',
        'replacement_char': 65533,
        'name': 'error_replace_default'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Error handling 'replace' with a custom replacement character
    input_dict = {
        'input': np.array([b'start\xfeend', b'another'], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'replace',
        'replacement_char': 35,
        'name': 'error_replace_custom'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Error handling 'ignore'
    input_dict = {
        'input': np.array([b'good\xed\xa0\x80morning', b'bad'], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'ignore',
        'replacement_char': 65533,
        'name': 'error_ignore'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: UTF-16-BE encoding
    input_dict = {
        'input': np.array([s.encode('utf-16be') for s in ('hello', 'G\xf6\xf6dnight')], dtype=object),
        'input_encoding': 'UTF-16-BE',
        'errors': 'strict',
        'replacement_char': 65533,
        'name': 'utf16_be'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Input with empty strings
    input_dict = {
        'input': np.array([b'first', b'', b'third'], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'strict',
        'replacement_char': 65533,
        'name': 'empty_strings'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: UTF-32-BE encoding (fixed from UTF-32-LE)
    input_dict = {
        'input': np.array([s.encode('utf-32be') for s in ('hello', '\U0001f60a')], dtype=object),
        'input_encoding': 'UTF-32-BE',
        'errors': 'strict',
        'replacement_char': 65533,
        'name': 'utf32_be'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D array with strings of different byte lengths
    input_dict = {
        'input': np.array([[b'short', b'a much longer string'], [b'another long one', b'tiny']], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'strict',
        'replacement_char': 65533,
        'name': 'ragged_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: An empty input tensor
    input_dict = {
        'input': np.array([], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'strict',
        'replacement_char': 65533,
        'name': 'empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strings.unicode_split"] = tf_strings_unicode_split_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strings.unicode_split' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.unicode_split'.")

check_valid('tf.strings.unicode_split', generated_inputs['tf.strings.unicode_split'], lib="tf", suffix=0)
