
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_split_inputs():
    list_of_inputs = []

    # Input 1: Basic whitespace split (using sep='' which mimics default behavior)
    input_dict = {
        'input': np.array(['  hello   world  ', 'a b c'], dtype=object),
        'sep': np.array('', dtype=object),
        'maxsplit': -1,
        'name': 'default_like_whitespace_split'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Specific space separator (creates empty strings for consecutive spaces)
    input_dict = {
        'input': np.array(['  hello   world  ', 'a b c'], dtype=object),
        'sep': np.array(' ', dtype=object),
        'maxsplit': -1,
        'name': 'specific_space_split'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Custom separator with no name
    input_dict = {
        'input': np.array(['apple,banana,cherry', 'dog,cat'], dtype=object),
        'sep': np.array(',', dtype=object),
        'maxsplit': -1,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive maxsplit
    input_dict = {
        'input': np.array(['one two three four', 'five six seven'], dtype=object),
        'sep': np.array(' ', dtype=object),
        'maxsplit': 2,
        'name': 'positive_maxsplit'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D input tensor
    input_dict = {
        'input': np.array([['a-b-c', 'd-e'], ['f-g', 'h-i-j-k']], dtype=object),
        'sep': np.array('-', dtype=object),
        'maxsplit': -1,
        'name': '2d_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-character separator
    input_dict = {
        'input': np.array(['1<>2<><>3', 'start<><>end'], dtype=object),
        'sep': np.array('<>', dtype=object),
        'maxsplit': -1,
        'name': 'multi_char_sep'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D (scalar) input tensor
    input_dict = {
        'input': np.array('scalar-string-to-split', dtype=object),
        'sep': np.array('-', dtype=object),
        'maxsplit': 1,
        'name': 'scalar_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: maxsplit = 0
    input_dict = {
        'input': np.array(['a,b,c', 'd,e'], dtype=object),
        'sep': np.array(',', dtype=object),
        'maxsplit': 0,
        'name': 'maxsplit_zero'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: No delimiter found in strings
    input_dict = {
        'input': np.array(['no_delimiter_here', 'another_one'], dtype=object),
        'sep': np.array('X', dtype=object),
        'maxsplit': -1,
        'name': 'no_delimiter_found'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty strings and consecutive delimiters
    input_dict = {
        'input': np.array(['', 'a;;c', ';d;', ' '], dtype=object),
        'sep': np.array(';', dtype=object),
        'maxsplit': -1,
        'name': 'empty_string_and_delimiters'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Unicode characters
    input_dict = {
        'input': np.array(['你好 世界', '你好 TensorFlow'], dtype=object),
        'sep': np.array(' ', dtype=object),
        'maxsplit': -1,
        'name': 'unicode_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Empty input tensor
    input_dict = {
        'input': np.array([], dtype=object),
        'sep': np.array(',', dtype=object),
        'maxsplit': -1,
        'name': 'empty_input_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strings.split"] = tf_strings_split_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strings.split' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.split'.")

check_valid('tf.strings.split', generated_inputs['tf.strings.split'], lib="tf", suffix=0)
