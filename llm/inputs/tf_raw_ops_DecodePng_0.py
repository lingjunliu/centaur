
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# Valid image data as bytes, generated using Pillow to ensure correctness.
# 1x1 red pixel PNG, 8-bit, RGB
PNG_8BIT_RGB = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc`p`\x00\x00\x00\x04\x00\x01\x99\x7f\r\xe4\x00\x00\x00\x00IEND\xaeB`\x82'
# 1x1 white pixel PNG, 8-bit, Grayscale
PNG_8BIT_GRAY = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00\x9a\x8d\x8f\x00\x00\x00\nIDATx\x9cc`\xf8\x0f\x00\x00\x02\x00\x01\x18\x8e\x9e[\x00\x00\x00\x00IEND\xaeB`\x82'
# 1x1 black pixel PNG, 16-bit, Grayscale (with correct CRC)
PNG_16BIT_GRAY = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x10\x00\x00\x00\x00\x92-\xde\xe0\x00\x00\x00\x0eIDATx\xda\xed\xc1\x01\x01\x00\x00\x00\xc2\xa0\xf7Om\x00\x01\x82\xfe\x01\xaf\x00\x00\x00\x00IEND\xaeB`\x82'
# 1x1 red pixel JPEG
JPEG_8BIT_RGB = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xdb\x00C\x01\t\t\t\x0c\x0b\x0c\x18\r\r\x182!\x1c!22222222222222222222222222222222222222222222222222\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11\x00\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00\x01\x02\x03\x11\x04\x05!1\x06\x12AQ\x07aq\x13"2\x81\x08\x14B\x91\xa1\xb1\xc1\t#3R\xf0\x15br\xd1\n\x16$4\xe1%\xf1\x17\x18\x19\x1a&\'()*56789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xfd\x17\x80\x01\x18\x00\x01\xff\xd9'
# 1x1 red pixel GIF
GIF_8BIT = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'

def get_raw_ops_decode_png_inputs():
    list_of_inputs = []

    # Input 1: Default PNG decoding (channels=0, dtype=uint8)
    input_dict_1 = {'contents': np.array(PNG_8BIT_RGB), 'channels': 0, 'dtype': np.uint8, 'name': 'default_png'}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Decode RGB PNG to grayscale (channels=1)
    input_dict_2 = {'contents': np.array(PNG_8BIT_RGB), 'channels': 1, 'dtype': np.uint8, 'name': 'rgb_to_gray'}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Decode grayscale PNG to RGB (channels=3)
    input_dict_3 = {'contents': np.array(PNG_8BIT_GRAY), 'channels': 3, 'dtype': np.uint8, 'name': 'gray_to_rgb'}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Decode RGB PNG to RGBA (channels=4)
    input_dict_4 = {'contents': np.array(PNG_8BIT_RGB), 'channels': 4, 'dtype': np.uint8, 'name': 'rgb_to_rgba'}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Decode JPEG with default channels (should be 3)
    input_dict_5 = {'contents': np.array(JPEG_8BIT_RGB), 'channels': 0, 'dtype': np.uint8, 'name': 'decode_jpeg'}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Decode JPEG to grayscale
    input_dict_6 = {'contents': np.array(JPEG_8BIT_RGB), 'channels': 1, 'dtype': np.uint8, 'name': 'jpeg_to_gray'}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Decode GIF with default channels (should be 3)
    input_dict_7 = {'contents': np.array(GIF_8BIT), 'channels': 0, 'dtype': np.uint8, 'name': 'decode_gif'}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Decode GIF to RGBA
    input_dict_8 = {'contents': np.array(GIF_8BIT), 'channels': 4, 'dtype': np.uint8, 'name': 'gif_to_rgba'}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Decode 16-bit PNG with dtype=uint16
    input_dict_9 = {'contents': np.array(PNG_16BIT_GRAY), 'channels': 0, 'dtype': np.uint16, 'name': 'decode_16bit_png'}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Decode 16-bit PNG to RGB, dtype=uint16
    input_dict_10 = {'contents': np.array(PNG_16BIT_GRAY), 'channels': 3, 'dtype': np.uint16, 'name': 'png16_to_rgb16'}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.DecodePng"] = get_raw_ops_decode_png_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DecodePng' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodePng'.")

check_valid('tf.raw_ops.DecodePng', generated_inputs['tf.raw_ops.DecodePng'], lib="tf", suffix=0)
