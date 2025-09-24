
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_strings_unicode_split_with_offsets_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor with UTF-8 encoding
    input_dict = {
        'input': np.array([s.encode('utf8') for s in ["hello", "world", "G\xf6\xf6dnight"]], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'replace',
        'replacement_char': 65533,
        'name': 'basic_utf8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar tensor input
    input_dict = {
        'input': np.array(u"Scalar string \U0001f60a".encode('utf8'), dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'strict',
        'replacement_char': 65533,
        'name': 'scalar_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor with empty strings
    input_dict = {
        'input': np.array([[b'first', b''], [b'third', b'fourth']], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'strict',
        'replacement_char': 65533,
        'name': '2d_with_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: UTF-16-BE encoding
    input_dict = {
        'input': np.array([s.encode('utf-16-be') for s in ["你好", "世界"]], dtype=object),
        'input_encoding': 'UTF-16-BE',
        'errors': 'strict',
        'replacement_char': 65533,
        'name': 'utf16_be_encoding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: UTF-32-BE encoding (changed from invalid UTF-32-LE)
    input_dict = {
        'input': np.array([s.encode('utf-32-be') for s in ["€", "Test"]], dtype=object),
        'input_encoding': 'UTF-32-BE',
        'errors': 'strict',
        'replacement_char': 65533,
        'name': 'utf32_be_encoding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Error handling 'replace' with custom replacement character
    input_dict = {
        'input': np.array([b'start\xff\xfeend'], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'replace',
        'replacement_char': 35,  # '#' character
        'name': 'error_replace_custom'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Error handling 'ignore'
    input_dict = {
        'input': np.array([b'start\xff\xfeend'], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'ignore',
        'replacement_char': 65533,
        'name': 'error_ignore'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with only one element
    input_dict = {
        'input': np.array([b'\xf0\x9f\x98\x8a'], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'replace',
        'replacement_char': 65533,
        'name': 'single_element'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All empty strings
    input_dict = {
        'input': np.array([b'', b''], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'strict',
        'replacement_char': 65533,
        'name': 'all_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D tensor input
    input_dict = {
        'input': np.array([[[b'a', b'b'], [b'c', b'd']], [[b'e', b'f'], [b'g', b'h']]], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'replace',
        'replacement_char': 65533,
        'name': '3d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Mixed valid and invalid strings with 'ignore'
    input_dict = {
        'input': np.array([b'valid', b'\xc3\x28', b'also_valid', b'\xa0\xa1'], dtype=object),
        'input_encoding': 'UTF-8',
        'errors': 'ignore',
        'replacement_char': 65533,
        'name': 'mixed_validity_ignore'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strings.unicode_split_with_offsets"] = tf_strings_unicode_split_with_offsets_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strings.unicode_split_with_offsets' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.unicode_split_with_offsets'.")

check_valid('tf.strings.unicode_split_with_offsets', generated_inputs['tf.strings.unicode_split_with_offsets'], lib="tf", suffix=0)
