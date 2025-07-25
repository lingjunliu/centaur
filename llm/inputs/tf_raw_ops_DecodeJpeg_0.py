
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_decode_jpeg_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.DecodeJpeg function.
    """
    # Using a minimal, known-valid 1x1 black pixel JPEG bytestring.
    # The previous errors were due to malformed JPEG data. This bytestring
    # is verified to be a valid JPEG.
    jpeg_bytes = (
        b'\xff\xd8\xff\xdb\x00\x43\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01'
        b'\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01'
        b'\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01'
        b'\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01'
        b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14'
        b'\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'\x00\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x10\xff\xd9'
    )
    contents_tensor = np.array(jpeg_bytes, dtype=object)

    # A truncated version for testing recovery.
    truncated_contents_tensor = np.array(jpeg_bytes[:-5], dtype=object)
    list_of_inputs = []

    # Input 1: Default decode
    input_dict_1 = {
        'contents': contents_tensor,
        'channels': 0,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'default_decode'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Grayscale output (channels=1)
    input_dict_2 = {
        'contents': contents_tensor,
        'channels': 1,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'grayscale_decode'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: RGB output (channels=3)
    input_dict_3 = {
        'contents': contents_tensor,
        'channels': 3,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'rgb_decode'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Downscale by factor of 2 (for a 1x1 image, this is allowed but won't change size)
    input_dict_4 = {
        'contents': contents_tensor,
        'channels': 0,
        'ratio': 2,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'ratio_2_decode'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Recover a truncated image
    input_dict_5 = {
        'contents': truncated_contents_tensor,
        'channels': 0,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': True,
        'acceptable_fraction': 0.5,
        'dct_method': '',
        'name': 'recover_truncated'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Turn off fancy upscaling
    input_dict_6 = {
        'contents': contents_tensor,
        'channels': 3,
        'ratio': 1,
        'fancy_upscaling': False,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'no_fancy_upscaling'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Use INTEGER_FAST DCT method
    input_dict_7 = {
        'contents': contents_tensor,
        'channels': 0,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': 'INTEGER_FAST',
        'name': 'fast_dct'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Use INTEGER_ACCURATE DCT method
    input_dict_8 = {
        'contents': contents_tensor,
        'channels': 0,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': 'INTEGER_ACCURATE',
        'name': 'accurate_dct'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Combination of downscaling and grayscale
    input_dict_9 = {
        'contents': contents_tensor,
        'channels': 1,
        'ratio': 4,
        'fancy_upscaling': False,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'combo_grayscale_downscale'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Combination with truncated recovery
    input_dict_10 = {
        'contents': truncated_contents_tensor,
        'channels': 3,
        'ratio': 8,
        'fancy_upscaling': True,
        'try_recover_truncated': True,
        'acceptable_fraction': 0.1,
        'dct_method': 'INTEGER_FAST',
        'name': 'combo_recover_fast'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.DecodeJpeg"] = get_tf_raw_ops_decode_jpeg_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DecodeJpeg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeJpeg'.")

check_valid('tf.raw_ops.DecodeJpeg', generated_inputs['tf.raw_ops.DecodeJpeg'], lib="tf", suffix=0)
