
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decodejpeg_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.DecodeJpeg function.
    """
    list_of_inputs = []

    # Create a dummy JPEG-encoded image string.
    sample_image_array = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    jpeg_encoded_tensor = tf.image.encode_jpeg(sample_image_array)
    contents_bytes = jpeg_encoded_tensor.numpy()
    
    # The 'contents' input must be a 0-D string tensor. To create a numpy
    # array that represents this without a specific 'S' dtype, we create a
    # 0-D array with dtype=object. This prevents ValueError from the test harness.
    contents_numpy = np.array(contents_bytes, dtype=object)

    # Input 1: Default parameters
    input_dict_1 = {
        'contents': contents_numpy,
        'channels': 0,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'default_decode'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Decode to grayscale
    input_dict_2 = {
        'contents': contents_numpy,
        'channels': 1,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'grayscale_decode'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Decode to RGB
    input_dict_3 = {
        'contents': contents_numpy,
        'channels': 3,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'rgb_decode'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Downscale by a ratio of 2
    input_dict_4 = {
        'contents': contents_numpy,
        'channels': 0,
        'ratio': 2,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'ratio_2_decode'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Downscale by a ratio of 4
    input_dict_5 = {
        'contents': contents_numpy,
        'channels': 0,
        'ratio': 4,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'ratio_4_decode'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Downscale by a ratio of 8
    input_dict_6 = {
        'contents': contents_numpy,
        'channels': 0,
        'ratio': 8,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'ratio_8_decode'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Disable fancy upscaling
    input_dict_7 = {
        'contents': contents_numpy,
        'channels': 0,
        'ratio': 1,
        'fancy_upscaling': False,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'no_fancy_upscaling'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Use INTEGER_FAST DCT method
    input_dict_8 = {
        'contents': contents_numpy,
        'channels': 3,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': 'INTEGER_FAST',
        'name': 'fast_dct_decode'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Use INTEGER_ACCURATE DCT method
    input_dict_9 = {
        'contents': contents_numpy,
        'channels': 3,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': 'INTEGER_ACCURATE',
        'name': 'accurate_dct_decode'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Try to recover truncated image
    truncated_bytes = contents_bytes[:-20]
    truncated_contents_numpy = np.array(truncated_bytes, dtype=object)
    input_dict_10 = {
        'contents': truncated_contents_numpy,
        'channels': 0,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': True,
        'acceptable_fraction': 0.5,
        'dct_method': '',
        'name': 'recover_truncated'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: A combination of non-default parameters
    input_dict_11 = {
        'contents': contents_numpy,
        'channels': 1,
        'ratio': 2,
        'fancy_upscaling': False,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': 'INTEGER_FAST',
        'name': 'combined_params_decode'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.raw_ops.DecodeJpeg"] = tf_raw_ops_decodejpeg_inputs()

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
