
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import struct

def tf_raw_ops_decode_bmp_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.DecodeBmp function.
    """

    def _create_bmp(width, height, pixels):
        """
        Creates a 24-bit BMP byte string.
        pixels is a list of (R, G, B) tuples, row by row, starting from top-left.
        """
        row_size_unpadded = width * 3
        padding = (4 - (row_size_unpadded % 4)) % 4
        row_size_padded = row_size_unpadded + padding
        image_data_size = row_size_padded * height
        file_header_size = 14
        dib_header_size = 40
        file_size = file_header_size + dib_header_size + image_data_size

        file_header = b'BM'
        file_header += struct.pack('<I', file_size)
        file_header += b'\x00\x00\x00\x00'
        file_header += struct.pack('<I', file_header_size + dib_header_size)
        dib_header = struct.pack('<I', dib_header_size)
        dib_header += struct.pack('<i', width)
        dib_header += struct.pack('<i', height)
        dib_header += struct.pack('<H', 1)
        dib_header += struct.pack('<H', 24)
        dib_header += struct.pack('<I', 0)
        dib_header += struct.pack('<I', image_data_size)
        dib_header += struct.pack('<i', 0)
        dib_header += struct.pack('<i', 0)
        dib_header += struct.pack('<I', 0)
        dib_header += struct.pack('<I', 0)
        
        pixel_data = b''
        for y in range(height - 1, -1, -1):
            row_data = b''
            for x in range(width):
                r, g, b = pixels[y * width + x]
                row_data += struct.pack('BBB', b, g, r)
            row_data += b'\x00' * padding
            pixel_data += row_data
            
        return file_header + dib_header + pixel_data

    list_of_inputs = []

    bmp_1x1_blue = _create_bmp(1, 1, [(0, 0, 255)])
    bmp_1x1_red = _create_bmp(1, 1, [(255, 0, 0)])
    bmp_1x1_green = _create_bmp(1, 1, [(0, 255, 0)])
    bmp_2x1_rg = _create_bmp(2, 1, [(255, 0, 0), (0, 255, 0)])
    bmp_2x2_multi = _create_bmp(2, 2, [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)])
    bmp_1x2_bw = _create_bmp(1, 2, [(0, 0, 0), (255, 255, 255)])

    # Input 1: Default channels (0), 1x1 blue pixel
    input_dict = {
        'contents': np.array(bmp_1x1_blue, dtype=object),
        'channels': 0,
        'name': 'decode_blue_default_channels'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3 channels, 1x1 red pixel
    input_dict = {
        'contents': np.array(bmp_1x1_red, dtype=object),
        'channels': 3,
        'name': 'decode_red_rgb'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4 channels, 1x1 green pixel
    input_dict = {
        'contents': np.array(bmp_1x1_green, dtype=object),
        'channels': 4,
        'name': 'decode_green_rgba'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Default channels (0), 2x1 red-green pixels
    input_dict = {
        'contents': np.array(bmp_2x1_rg, dtype=object),
        'channels': 0,
        'name': 'decode_2x1_default'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3 channels, 2x2 multi-color pixels
    input_dict = {
        'contents': np.array(bmp_2x2_multi, dtype=object),
        'channels': 3,
        'name': 'decode_2x2_rgb'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4 channels, 2x2 multi-color pixels
    input_dict = {
        'contents': np.array(bmp_2x2_multi, dtype=object),
        'channels': 4,
        'name': 'decode_2x2_rgba'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Default channels (0), 1x2 black-white pixels
    input_dict = {
        'contents': np.array(bmp_1x2_bw, dtype=object),
        'channels': 0,
        'name': 'decode_1x2_bw_default'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3 channels, 1x2 black-white pixels
    input_dict = {
        'contents': np.array(bmp_1x2_bw, dtype=object),
        'channels': 3,
        'name': 'decode_1x2_bw_rgb'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4 channels, 1x1 blue pixel
    input_dict = {
        'contents': np.array(bmp_1x1_blue, dtype=object),
        'channels': 4,
        'name': 'decode_blue_rgba'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3 channels, 2x1 red-green pixels
    input_dict = {
        'contents': np.array(bmp_2x1_rg, dtype=object),
        'channels': 3,
        'name': 'decode_2x1_rgb'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4 channels, 2x1 red-green pixels with an empty name
    input_dict = {
        'contents': np.array(bmp_2x1_rg, dtype=object),
        'channels': 4,
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Default channels with a long name
    input_dict = {
        'contents': np.array(bmp_1x1_red, dtype=object),
        'channels': 0,
        'name': 'a_very_long_and_descriptive_name_for_the_operation_to_test_string_handling'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DecodeBmp"] = tf_raw_ops_decode_bmp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DecodeBmp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeBmp'.")

check_valid('tf.raw_ops.DecodeBmp', generated_inputs['tf.raw_ops.DecodeBmp'], lib="tf", suffix=0)
