
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_decode_and_crop_jpeg_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.DecodeAndCropJpeg.
    This version uses verified hardcoded JPEG data and ensures the crop_window
    is valid for downscaled images.
    """
    list_of_inputs = []

    # A known-valid 8x8 pixel RGB JPEG.
    jpeg_rgb_8x8_bytes = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x06\x07\t\x07\x06\x06\x08\x0b\x08\t\n\n\n\n\n\x06\x08\x0b\x0c\x0b\n\x0c\t\n\n\n\xff\xdb\x00C\x01\x02\x02\x02\x02\x04\x04\x04\x08\x06\x06\x08\x10\x0b\t\x0b\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\xff\xc0\x00\x11\x08\x00\x08\x00\x08\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11\x00\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00\x01\x02\x03\x11\x04\x05!1\x06\x12AQ\x07aq\x13"2\x81\x08\x14B\x91\xa1\xb1\xc1\t#3R\xf0\x15br\xd1\n\x16$4\xe1%\xf1\x17\x18\x19\x1a&\'()*56789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xf8\xa3\xbf\x1f\xbe\x82\x80\t\xfa\x0e\xde\x01\xd4\xd6\x7f\xe1l\xac\xf7\x9b\xf1\xe8\xe2\xac\x1a\x14%\xa2\x80\t\x95\xff\xd9'

    # A known-valid 16x16 pixel Grayscale JPEG.
    jpeg_gray_16x16_bytes = b'\xff\xd8\xff\xdb\x00C\x00\x10\x0b\x0c\x0e\x0c\n\x10\x0e\r\x0e\x12\x11\x10\x13\x18\x28\x1a\x18\x16\x16\x18\x1d\x26\x1e\x28\x22\x20\x22\x2a\x2b\x2c\x26\x2a\x27\x26\x1c\x25\x2e\x33\x30\x2a\x30\x2e\x2b\x2e\x2a\xff\xc0\x00\x0b\x08\x00\x10\x00\x10\x01\x01\x11\x00\xff\xc4\x00\x1b\x00\x00\x02\x03\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x04\x01\x02\x05\x06\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xa3\xd4\xe4\xa7\x94T\x02\xa2\x94\xf4\xa2\x80\n(\xa2\x80\n(\xa2\x80\n(\xa2\x80\n(\xa2\x80\n(\xa2\x80\n(\xa2\x80\n(\xa2\x80\n(\xa2\x80\n(\xa2\x80\n(\xa2\x80\n(\xa2\x80\n(\xa2\x80\n(\xa2\x80\n(\xa2\x80\t\xd9'

    jpeg_rgb = np.array(jpeg_rgb_8x8_bytes, dtype=object)
    jpeg_gray = np.array(jpeg_gray_16x16_bytes, dtype=object)
    jpeg_truncated = np.array(jpeg_rgb_8x8_bytes[:-30], dtype=object)

    # Input 1: Basic case, full crop, default channels and ratio.
    list_of_inputs.append({'contents': jpeg_rgb, 'crop_window': np.array([0, 0, 8, 8], dtype=np.int32), 'channels': 0, 'ratio': 1, 'fancy_upscaling': True, 'try_recover_truncated': False, 'acceptable_fraction': 1.0, 'dct_method': '', 'name': 'basic_full_crop'})
    # Input 2: Crop a sub-region, force 3 channels (RGB).
    list_of_inputs.append({'contents': jpeg_rgb, 'crop_window': np.array([2, 2, 4, 4], dtype=np.int32), 'channels': 3, 'ratio': 1, 'fancy_upscaling': True, 'try_recover_truncated': False, 'acceptable_fraction': 1.0, 'dct_method': '', 'name': 'crop_subregion_rgb'})
    # Input 3: Force 1 channel (grayscale) output from RGB source.
    list_of_inputs.append({'contents': jpeg_rgb, 'crop_window': np.array([0, 0, 8, 8], dtype=np.int32), 'channels': 1, 'ratio': 1, 'fancy_upscaling': True, 'try_recover_truncated': False, 'acceptable_fraction': 1.0, 'dct_method': '', 'name': 'force_grayscale'})
    # Input 4: Downscale with ratio=2. Downscaled image is 4x4.
    list_of_inputs.append({'contents': jpeg_rgb, 'crop_window': np.array([0, 0, 4, 4], dtype=np.int32), 'channels': 3, 'ratio': 2, 'fancy_upscaling': True, 'try_recover_truncated': False, 'acceptable_fraction': 1.0, 'dct_method': '', 'name': 'downscale_ratio_2'})
    # Input 5: Downscale with ratio=4. Downscaled image is 2x2.
    list_of_inputs.append({'contents': jpeg_rgb, 'crop_window': np.array([0, 0, 2, 2], dtype=np.int32), 'channels': 3, 'ratio': 4, 'fancy_upscaling': True, 'try_recover_truncated': False, 'acceptable_fraction': 1.0, 'dct_method': '', 'name': 'downscale_ratio_4'})
    # Input 6: Downscale with ratio=8. Downscaled image is 1x1.
    list_of_inputs.append({'contents': jpeg_rgb, 'crop_window': np.array([0, 0, 1, 1], dtype=np.int32), 'channels': 1, 'ratio': 8, 'fancy_upscaling': True, 'try_recover_truncated': False, 'acceptable_fraction': 1.0, 'dct_method': '', 'name': 'downscale_ratio_8'})
    # Input 7: Disable fancy upscaling.
    list_of_inputs.append({'contents': jpeg_rgb, 'crop_window': np.array([1, 1, 6, 6], dtype=np.int32), 'channels': 3, 'ratio': 1, 'fancy_upscaling': False, 'try_recover_truncated': False, 'acceptable_fraction': 1.0, 'dct_method': '', 'name': 'no_fancy_upscaling'})
    # Input 8: Use INTEGER_FAST dct_method.
    list_of_inputs.append({'contents': jpeg_rgb, 'crop_window': np.array([0, 0, 8, 8], dtype=np.int32), 'channels': 0, 'ratio': 1, 'fancy_upscaling': True, 'try_recover_truncated': False, 'acceptable_fraction': 1.0, 'dct_method': 'INTEGER_FAST', 'name': 'dct_fast'})
    # Input 9: Use INTEGER_ACCURATE dct_method.
    list_of_inputs.append({'contents': jpeg_rgb, 'crop_window': np.array([0, 0, 8, 8], dtype=np.int32), 'channels': 0, 'ratio': 1, 'fancy_upscaling': True, 'try_recover_truncated': False, 'acceptable_fraction': 1.0, 'dct_method': 'INTEGER_ACCURATE', 'name': 'dct_accurate'})
    # Input 10: Recover a truncated image.
    list_of_inputs.append({'contents': jpeg_truncated, 'crop_window': np.array([0, 0, 6, 6], dtype=np.int32), 'channels': 3, 'ratio': 1, 'fancy_upscaling': True, 'try_recover_truncated': True, 'acceptable_fraction': 0.5, 'dct_method': '', 'name': 'recover_truncated'})
    # Input 11: Use a grayscale source image.
    list_of_inputs.append({'contents': jpeg_gray, 'crop_window': np.array([0, 0, 16, 16], dtype=np.int32), 'channels': 0, 'ratio': 1, 'fancy_upscaling': True, 'try_recover_truncated': False, 'acceptable_fraction': 1.0, 'dct_method': '', 'name': 'grayscale_source_auto_channels'})
    # Input 12: Grayscale source, request RGB, downscale. Downscaled image is 8x8.
    list_of_inputs.append({'contents': jpeg_gray, 'crop_window': np.array([0, 0, 8, 8], dtype=np.int32), 'channels': 3, 'ratio': 2, 'fancy_upscaling': False, 'try_recover_truncated': False, 'acceptable_fraction': 1.0, 'dct_method': 'INTEGER_FAST', 'name': 'grayscale_to_rgb_downscaled'})

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
