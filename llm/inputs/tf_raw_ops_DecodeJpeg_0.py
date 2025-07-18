
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import base64

def tf_raw_ops_decodejpeg_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.DecodeJpeg.
    """
    # A valid base64-encoded 2x2 red JPEG image. This is a minimal, valid JPEG.
    b64_jpeg = b'/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAACAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1VldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/KKKKAP/2Q=='
    valid_jpeg_bytes = base64.b64decode(b64_jpeg)
    
    # Create a scalar numpy array with dtype=object to hold the bytes
    # to avoid specific string dtype issues like dtype('S...').
    valid_jpeg_contents = np.array(valid_jpeg_bytes, dtype=object)
    
    list_of_inputs = []

    # Input 1: Default parameters
    list_of_inputs.append({
        'contents': copy.deepcopy(valid_jpeg_contents),
        'channels': 0,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'default_decode'
    })

    # Input 2: Grayscale output
    list_of_inputs.append({
        'contents': copy.deepcopy(valid_jpeg_contents),
        'channels': 1,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'grayscale_decode'
    })

    # Input 3: RGB output
    list_of_inputs.append({
        'contents': copy.deepcopy(valid_jpeg_contents),
        'channels': 3,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'rgb_decode'
    })

    # Input 4: Downscaling with ratio=2 (Note: image is 2x2, so ratio > 1 will result in 1x1)
    list_of_inputs.append({
        'contents': copy.deepcopy(valid_jpeg_contents),
        'channels': 0,
        'ratio': 2,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'ratio2_decode'
    })

    # Input 5: No fancy upscaling
    list_of_inputs.append({
        'contents': copy.deepcopy(valid_jpeg_contents),
        'channels': 0,
        'ratio': 1,
        'fancy_upscaling': False,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'no_fancy_upscaling'
    })

    # Input 6: INTEGER_FAST DCT method
    list_of_inputs.append({
        'contents': copy.deepcopy(valid_jpeg_contents),
        'channels': 0,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': 'INTEGER_FAST',
        'name': 'fast_dct'
    })

    # Input 7: INTEGER_ACCURATE DCT method
    list_of_inputs.append({
        'contents': copy.deepcopy(valid_jpeg_contents),
        'channels': 0,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': 'INTEGER_ACCURATE',
        'name': 'accurate_dct'
    })

    # Input 8: Combination of parameters
    list_of_inputs.append({
        'contents': copy.deepcopy(valid_jpeg_contents),
        'channels': 1,
        'ratio': 2,
        'fancy_upscaling': False,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': 'INTEGER_FAST',
        'name': 'combo_grayscale_fast_ratio2'
    })
    
    # Input 9: Another RGB combo
    list_of_inputs.append({
        'contents': copy.deepcopy(valid_jpeg_contents),
        'channels': 3,
        'ratio': 1,
        'fancy_upscaling': False,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': 'INTEGER_ACCURATE',
        'name': 'combo_rgb_accurate_nofancy'
    })

    # Input 10: Another grayscale combo
    list_of_inputs.append({
        'contents': copy.deepcopy(valid_jpeg_contents),
        'channels': 1,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'grayscale_fancy'
    })

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
