
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# Assume generated_inputs is pre-initialized
# generated_inputs = {}

def tf_raw_ops_decode_base64_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.DecodeBase64 function.
    """
    list_of_inputs = []

    # Input 1: Scalar tensor with a common base64 string
    input_dict = {
        'input': np.array('SGVsbG8sIFdvcmxkIQ==', dtype=object),
        'name': 'scalar_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor (vector) of base64 strings
    input_dict = {
        'input': np.array(['dGVuc29yZmxvdw==', 'bWFuY2hpbmUgbGVhcm5pbmc='], dtype=object),
        'name': '1d_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor (matrix) of base64 strings
    input_dict = {
        'input': np.array([['QUk=', 'UHl0aG9u'], ['RGVlcCBMZWFybmluZw==', 'TnVtUHk=']], dtype=object),
        'name': '2d_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Input with web-safe characters (- and _)
    # b'\xfb\xff\xbf' encodes to '-_-_' in web-safe base64
    input_dict = {
        'input': np.array(['-_-_', 'YWJjZA'], dtype=object),
        'name': 'web_safe_chars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Input without padding (padding is optional)
    # 'QUk=' ('AI') without padding is 'QUk'
    # 'SGVsbG8=' ('Hello') without padding is 'SGVsbG8'
    input_dict = {
        'input': np.array(['QUk', 'SGVsbG8'], dtype=object),
        'name': 'no_padding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Input with mixed padding and no-padding strings
    input_dict = {
        'input': np.array(['SGVsbG8', 'dGVuc29yZmxvdw=='], dtype=object),
        'name': 'mixed_padding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Input containing an empty string, which decodes to an empty string
    input_dict = {
        'input': np.array(['', 'SGVsbG8sIFdvcmxkIQ=='], dtype=object),
        'name': 'with_empty_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar tensor with an empty string
    input_dict = {
        'input': np.array('', dtype=object),
        'name': 'scalar_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty 1D tensor
    input_dict = {
        'input': np.array([], dtype=object),
        'name': 'empty_1d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D tensor
    input_dict = {
        'input': np.array([[['QUk=']], [['UHl0aG9u']]], dtype=object),
        'name': '3d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: A longer string
    long_str = 'VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZw=='
    input_dict = {
        'input': np.array([long_str], dtype=object),
        'name': 'long_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

# Assume generated_inputs is pre-initialized
# For example:
# generated_inputs = {}
generated_inputs["tf.raw_ops.DecodeBase64"] = tf_raw_ops_decode_base64_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DecodeBase64' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeBase64'.")

check_valid('tf.raw_ops.DecodeBase64', generated_inputs['tf.raw_ops.DecodeBase64'], lib="tf", suffix=0)
