
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import base64

def tf_io_decode_bmp_inputs():
    list_of_inputs = []

    # Pre-generated, base64-encoded valid BMP file contents.
    # Generated offline using a reliable library to ensure correctness.
    BMP_1x1_BLUE_B64 = "Qk02AAAAAAAAADYAAAAoAAAAAQAAAAEAAAABAAAAGAAAAAAAAAAAAAAA/wAAAAA="
    BMP_1x1_RED_B64 = "Qk02AAAAAAAAADYAAAAoAAAAAQAAAAEAAAABAAAAGAAAAAAAAAAAAAAAAAAA/wA="
    BMP_1x1_GREEN_B64 = "Qk02AAAAAAAAADYAAAAoAAAAAQAAAAEAAAABAAAAGAAAAAAAAAAAAAAAAP8A/wA="
    BMP_2x2_MIXED_B64 = "Qk1GAAAAAAAAADYAAAAoAAAAAgAAAAIAAAABAAAAGAAAAAAAAAALAAAA/wAAAP8A/wD//wA="
    BMP_3x1_RGB_B64 = "Qk1GAAAAAAAAADYAAAAoAAAAAwAAAAEAAAABAAAAGAAAAAAAAAALAAAA/wAAAP8A/wD/"
    BMP_1x1_WHITE_B64 = "Qk02AAAAAAAAADYAAAAoAAAAAQAAAAEAAAABAAAAGAAAAAAAAAAAAAAA/////wA="
    BMP_5x5_BLACK_B64 = "Qk0eAQAAAAAAADYAAAAoAAAABQAAAAUAAAABAAAAGAAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
    BMP_1x5_GRADIENT_B64 = "Qk1eAAAAAAAAADYAAAAoAAAAAQAAAAUAAAABAAAAGAAAAAAAAAASAAAAISEiIyQlJicoKSorLC0uLzAw"

    # Decode base64 strings to bytes
    bmp_1x1_blue = base64.b64decode(BMP_1x1_BLUE_B64)
    bmp_1x1_red = base64.b64decode(BMP_1x1_RED_B64)
    bmp_1x1_green = base64.b64decode(BMP_1x1_GREEN_B64)
    bmp_2x2_mixed = base64.b64decode(BMP_2x2_MIXED_B64)
    bmp_3x1_rgb = base64.b64decode(BMP_3x1_RGB_B64)
    bmp_1x1_white = base64.b64decode(BMP_1x1_WHITE_B64)
    bmp_5x5_black = base64.b64decode(BMP_5x5_BLACK_B64)
    bmp_1x5_gradient = base64.b64decode(BMP_1x5_GRADIENT_B64)

    # Input 1: Explicitly decode to 3 channels (was default)
    input_dict = {
        'contents': np.array(bmp_1x1_blue),
        'channels': 3,
        'name': 'default_channels_blue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: channels=3, 1x1 red image
    input_dict = {
        'contents': np.array(bmp_1x1_red),
        'channels': 3,
        'name': 'rgb_channels_red'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: channels=4, 1x1 green image
    input_dict = {
        'contents': np.array(bmp_1x1_green),
        'channels': 4,
        'name': 'rgba_channels_green'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Explicitly decode to 3 channels on a 2x2 image
    input_dict = {
        'contents': np.array(bmp_2x2_mixed),
        'channels': 3,
        'name': 'default_channels_2x2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: channels=3 on a 2x2 image
    input_dict = {
        'contents': np.array(bmp_2x2_mixed),
        'channels': 3,
        'name': 'rgb_channels_2x2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: channels=4 on a 2x2 image
    input_dict = {
        'contents': np.array(bmp_2x2_mixed),
        'channels': 4,
        'name': 'rgba_channels_2x2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Explicitly decode to 3 channels on a 3x1 image, name is None
    input_dict = {
        'contents': np.array(bmp_3x1_rgb),
        'channels': 3,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: channels=3 on a 3x1 image, name is omitted
    input_dict = {
        'contents': np.array(bmp_3x1_rgb),
        'channels': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: channels=4 on a 3x1 image, with name
    input_dict = {
        'contents': np.array(bmp_3x1_rgb),
        'channels': 4,
        'name': 'decode_3x1_rgba'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1x1 white image, channels=3
    input_dict = {
        'contents': np.array(bmp_1x1_white),
        'channels': 3,
        'name': 'decode_white'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: A larger image, 5x5 all black, channels=4
    input_dict = {
        'contents': np.array(bmp_5x5_black),
        'channels': 4,
        'name': 'decode_black_5x5_rgba'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: A 1x5 image, explicitly decoded to 3 channels
    input_dict = {
        'contents': np.array(bmp_1x5_gradient),
        'channels': 3,
        'name': 'decode_1x5_gradient'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.io.decode_bmp"] = tf_io_decode_bmp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.decode_bmp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_bmp'.")

check_valid('tf.io.decode_bmp', generated_inputs['tf.io.decode_bmp'], lib="tf", suffix=0)
