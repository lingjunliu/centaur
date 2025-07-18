
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_decodeimage_inputs():
    # Minimal but valid image byte strings generated programmatically to ensure correctness.
    # 1x1 transparent PNG
    PNG_BYTES = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    # 1x1 black JPEG
    JPEG_BYTES = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c \x24\x2e\x27\x20\x22\x2c\x23\x1c\x1c(\x37\x2f\x2c\x30\x31\x34\x34\x34\x1f\x27\x39\x3d\x38\x32\x3c\x2e\x33\x34\x32\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xd2\xc4\xff\xd9'
    # 1x1 black non-animated GIF
    GIF_BYTES = b'GIF87a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
    # 1x1 red 24-bit BMP
    BMP_BYTES = b'BM\x3a\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x00\x00\x00'
    # 2-frame 1x1 animated GIF (black, then red)
    ANIMATED_GIF_BYTES = b'GIF89a\x01\x00\x01\x00\x80\x01\x00\x00\x00\x00\xff\x00\x00\xff\xff\xff!\xf9\x04\x01d\x00\x01\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02L\x01\x00!\xf9\x04\x01d\x00\x02\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
    # 1x1 green lossy WebP
    WEBP_BYTES = b'RIFF \x00\x00\x00WEBPVP8\x12\x00\x00\x00\x9d\x01*\x01\x00\x01\x90\x01\x02\x00\x02\x90\xfe\x0f\x00'
    
    list_of_inputs = []

    # Input 1: PNG, default settings
    input_dict_1 = {
        'contents': np.array(PNG_BYTES, dtype=object),
        'channels': 0,
        'dtype': np.uint8,
        'expand_animations': True,
        'name': 'decode_png_default'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: JPEG, 3 channels, float32 dtype
    input_dict_2 = {
        'contents': np.array(JPEG_BYTES, dtype=object),
        'channels': 3,
        'dtype': np.float32,
        'expand_animations': True,
        'name': 'decode_jpeg_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Non-animated GIF, uint16 dtype
    input_dict_3 = {
        'contents': np.array(GIF_BYTES, dtype=object),
        'channels': 0,
        'dtype': np.uint16,
        'expand_animations': True,
        'name': 'decode_gif_uint16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: BMP, 0 channels (auto-detect)
    input_dict_4 = {
        'contents': np.array(BMP_BYTES, dtype=object),
        'channels': 0,
        'dtype': np.uint8,
        'expand_animations': True,
        'name': 'decode_bmp_auto_channels'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Animated GIF, expand animations (4D output)
    input_dict_5 = {
        'contents': np.array(ANIMATED_GIF_BYTES, dtype=object),
        'channels': 0,
        'dtype': np.uint8,
        'expand_animations': True,
        'name': 'decode_animated_gif_expanded'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Animated GIF, do not expand animations (3D output, truncated)
    input_dict_6 = {
        'contents': np.array(ANIMATED_GIF_BYTES, dtype=object),
        'channels': 3,
        'dtype': np.uint8,
        'expand_animations': False,
        'name': 'decode_animated_gif_truncated'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: PNG, force 4 channels (RGBA), don't expand
    input_dict_7 = {
        'contents': np.array(PNG_BYTES, dtype=object),
        'channels': 4,
        'dtype': np.uint8,
        'expand_animations': False,
        'name': 'decode_png_rgba'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: WebP, default settings
    input_dict_8 = {
        'contents': np.array(WEBP_BYTES, dtype=object),
        'channels': 0,
        'dtype': np.uint8,
        'expand_animations': True,
        'name': 'decode_webp_default'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: JPEG, 1 channel (grayscale)
    input_dict_9 = {
        'contents': np.array(JPEG_BYTES, dtype=object),
        'channels': 1,
        'dtype': np.float32,
        'expand_animations': False,
        'name': 'decode_jpeg_grayscale_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: BMP, 3 channels, uint16
    input_dict_10 = {
        'contents': np.array(BMP_BYTES, dtype=object),
        'channels': 3,
        'dtype': np.uint16,
        'expand_animations': True,
        'name': 'decode_bmp_rgb_uint16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.DecodeImage"] = get_tf_raw_ops_decodeimage_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DecodeImage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeImage'.")

check_valid('tf.raw_ops.DecodeImage', generated_inputs['tf.raw_ops.DecodeImage'], lib="tf", suffix=0)
