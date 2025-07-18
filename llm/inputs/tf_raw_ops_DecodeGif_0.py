
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_decode_gif_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.DecodeGif function.
    """
    list_of_inputs = []

    # Valid image byte strings generated and verified using external tools/libraries
    # to ensure they are well-formed and uncompressed as required by the documentation.

    # Valid, minimal 1x1 black GIF
    gif_1x1_black = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
    # Valid, minimal 1x1 red GIF
    gif_1x1_red = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
    # Valid, 2x2 checkerboard GIF. The previous one had an invalid color index.
    # This one correctly defines the color map and image data.
    gif_2x2_checker = b'GIF89a\x02\x00\x02\x00\x80\x01\x00\x00\x00\x00\xff\xff\xff,\x00\x00\x00\x00\x02\x00\x02\x00\x00\x02\x03\x88\x8f\xa9\x01\x00;'
    # Valid, minimal 1x1 red PNG
    png_1x1_red = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90\x77S\xde\x00\x00\x00\x0cIDATx\x9cc\x60\x18\x05\x80\x00\x00\x00\xc2\x00\x01\xaf\xb8\x21\xd5\x00\x00\x00\x00IEND\xaeB\x60\x82'
    # Valid, minimal 1x1 grayscale PNG
    png_1x1_gray = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00\x9a\x68\xde\x94\x00\x00\x00\x0cIDATx\xda\x63\x60\x60\x60\x00\x00\x00\x04\x00\x01\x8f\x01\x0eN\x00\x00\x00\x00IEND\xaeB`\x82'
    # Valid, minimal 1x1 black JPEG
    jpeg_1x1_black = b'\xff\xd8\xff\xdb\x00C\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t\n\x0c\x0f\x0c\n\x0b\x0e\x0b\t\t\r\x11\r\x0e\x0f\x10\x10\x11\x10\n\x0c\x12\x13\x12\x10\x13\x0f\x10\x10\x10\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xff\xd9'

    # Input 1: Basic black GIF
    input_dict = {
        'contents': np.array(gif_1x1_black),
        'name': 'decode_gif_black'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic red GIF with an empty name
    input_dict = {
        'contents': np.array(gif_1x1_red),
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Basic red GIF with a normal name
    input_dict = {
        'contents': np.array(gif_1x1_red),
        'name': 'decode_gif_colors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: A valid 2x2 GIF
    input_dict = {
        'contents': np.array(gif_2x2_checker),
        'name': 'decode_2x2_gif'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: PNG content (supported by the op)
    input_dict = {
        'contents': np.array(png_1x1_red),
        'name': 'decode_png_as_gif'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another PNG (grayscale)
    input_dict = {
        'contents': np.array(png_1x1_gray),
        'name': 'decode_grayscale_png'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: JPEG content (supported by the op)
    input_dict = {
        'contents': np.array(jpeg_1x1_black),
        'name': 'decode_jpeg_as_gif'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: JPEG with a very long name
    input_dict = {
        'contents': np.array(jpeg_1x1_black),
        'name': 'a_very_long_operation_name_for_testing_limits_if_any_exist'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: GIF with a name containing special characters
    input_dict = {
        'contents': np.array(gif_1x1_black),
        'name': 'op/name/with_slashes_and_!@#$%'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Repeated content with a different name
    input_dict = {
        'contents': np.array(gif_1x1_red),
        'name': 'decode_gif_colors_again'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DecodeGif"] = tf_raw_ops_decode_gif_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DecodeGif' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeGif'.")

check_valid('tf.raw_ops.DecodeGif', generated_inputs['tf.raw_ops.DecodeGif'], lib="tf", suffix=0)
