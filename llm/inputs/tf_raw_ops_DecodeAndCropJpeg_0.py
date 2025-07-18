
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy


def get_tf_raw_ops_decode_and_crop_jpeg_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.DecodeAndCropJpeg function.
    """
    list_of_inputs = []

    # Helper to generate a dummy JPEG image content
    image_height = 128
    image_width = 128
    image_data = np.random.randint(0, 256, size=(image_height, image_width, 3), dtype=np.uint8)
    base_jpeg_contents = tf.image.encode_jpeg(image_data, format='rgb').numpy()
    contents_np = np.array(base_jpeg_contents, dtype=object)

    # --- Input 1: Basic case with default-like values ---
    input_dict = {
        'contents': contents_np,
        'crop_window': np.array([10, 10, 50, 50], dtype=np.int32),
        'channels': 0,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'default_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 2: Grayscale output ---
    input_dict = {
        'contents': contents_np,
        'crop_window': np.array([0, 0, 64, 64], dtype=np.int32),
        'channels': 1,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'grayscale_output'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 3: Explicit RGB output ---
    input_dict = {
        'contents': contents_np,
        'crop_window': np.array([0, 0, image_height, image_width], dtype=np.int32),
        'channels': 3,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'rgb_output'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # --- Input 4: Downscaling with ratio=2 ---
    # Effective image size is 128/2 = 64x64. Crop window must be within these bounds.
    input_dict = {
        'contents': contents_np,
        'crop_window': np.array([8, 8, 50, 50], dtype=np.int32),
        'channels': 0,
        'ratio': 2,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'downscale_ratio_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # --- Input 5: Downscaling with ratio=4 ---
    # Effective image size is 128/4 = 32x32.
    input_dict = {
        'contents': contents_np,
        'crop_window': np.array([4, 4, 20, 20], dtype=np.int32),
        'channels': 3,
        'ratio': 4,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'downscale_ratio_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # --- Input 6: Downscaling with ratio=8 ---
    # Effective image size is 128/8 = 16x16.
    input_dict = {
        'contents': contents_np,
        'crop_window': np.array([0, 0, 16, 16], dtype=np.int32),
        'channels': 1,
        'ratio': 8,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'downscale_ratio_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 7: No fancy upscaling ---
    input_dict = {
        'contents': contents_np,
        'crop_window': np.array([20, 20, 90, 90], dtype=np.int32),
        'channels': 3,
        'ratio': 1,
        'fancy_upscaling': False,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': '',
        'name': 'no_fancy_upscaling'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 8: Try to recover truncated image ---
    truncated_jpeg_contents = base_jpeg_contents[:-50]
    input_dict = {
        'contents': np.array(truncated_jpeg_contents, dtype=object),
        'crop_window': np.array([0, 0, 100, 100], dtype=np.int32),
        'channels': 0,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': True,
        'acceptable_fraction': 0.5,
        'dct_method': '',
        'name': 'recover_truncated'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 9: DCT method INTEGER_FAST ---
    # Effective image size is 128/2 = 64x64.
    input_dict = {
        'contents': contents_np,
        'crop_window': np.array([10, 10, 32, 32], dtype=np.int32),
        'channels': 3,
        'ratio': 2,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': 'INTEGER_FAST',
        'name': 'dct_fast'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 10: DCT method INTEGER_ACCURATE ---
    input_dict = {
        'contents': contents_np,
        'crop_window': np.array([5, 5, 110, 110], dtype=np.int32),
        'channels': 0,
        'ratio': 1,
        'fancy_upscaling': True,
        'try_recover_truncated': False,
        'acceptable_fraction': 1.0,
        'dct_method': 'INTEGER_ACCURATE',
        'name': 'dct_accurate'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 11: A combination of non-default values ---
    # Effective image size is 128/4 = 32x32.
    input_dict = {
        'contents': contents_np,
        'crop_window': np.array([0, 0, 16, 32], dtype=np.int32),
        'channels': 1,
        'ratio': 4,
        'fancy_upscaling': False,
        'try_recover_truncated': False,
        'acceptable_fraction': 0.99,
        'dct_method': 'INTEGER_FAST',
        'name': 'combo_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.DecodeAndCropJpeg"] = get_tf_raw_ops_decode_and_crop_jpeg_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DecodeAndCropJpeg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeAndCropJpeg'.")

check_valid('tf.raw_ops.DecodeAndCropJpeg', generated_inputs['tf.raw_ops.DecodeAndCropJpeg'], lib="tf", suffix=0)
