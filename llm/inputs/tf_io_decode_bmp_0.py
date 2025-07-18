
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import struct

def tf_io_decode_bmp_inputs():
    """
    Generates a list of valid inputs for tf.io.decode_bmp.
    """

    def create_bmp_24bpp(width, height, pixel_data_bgr):
        """Creates a valid 24-bpp BMP byte string."""
        padded_row_size = (width * 3 + 3) & ~3
        pixel_data_size = padded_row_size * height
        file_size = 54 + pixel_data_size
        
        # File Header
        header = struct.pack('<2sIHHI', b'BM', file_size, 0, 0, 54)
        # DIB Header
        header += struct.pack('<IiiHHIIIIII', 40, width, height, 1, 24, 0, pixel_data_size, 0, 0, 0, 0)
        
        padded_data = b''
        for i in range(height):
            row_start = i * width * 3
            row_end = row_start + width * 3
            row = pixel_data_bgr[row_start:row_end]
            padded_data += row + b'\x00' * (padded_row_size - len(row))

        return header + padded_data

    def create_bmp_32bpp(width, height, pixel_data_bgra):
        """Creates a valid 32-bpp BMP byte string (top-down)."""
        row_size = width * 4
        pixel_data_size = row_size * height
        file_size = 54 + pixel_data_size

        # File Header
        header = struct.pack('<2sIHHI', b'BM', file_size, 0, 0, 54)
        # DIB Header (negative height for top-down)
        header += struct.pack('<IiiHHIIIIII', 40, width, -height, 1, 32, 0, pixel_data_size, 0, 0, 0, 0)
        
        return header + pixel_data_bgra
        
    # --- Generate different valid BMP contents ---

    # Content 1: A 2x2 RGB image
    # Row 1 (bottom): Blue, Green
    # Row 0 (top):    Red, White
    pixels_2x2_bgr = (
        b'\xff\x00\x00' b'\x00\xff\x00'   # Bottom row: Blue, Green
        b'\x00\x00\xff' b'\xff\xff\xff'   # Top row: Red, White
    )
    contents_rgb = np.array(create_bmp_24bpp(2, 2, pixels_2x2_bgr))

    # Content 2: A 1x2 RGBA image
    # Row 0: transparent red, Row 1: semi-transparent blue
    pixels_1x2_bgra = (
        b'\x00\x00\xff\x00'  # BGRA for transparent red
        b'\xff\x00\x00\x80'  # BGRA for semi-transparent blue
    )
    contents_rgba = np.array(create_bmp_32bpp(1, 2, pixels_1x2_bgra))

    # Content 3: A 1x1 RGB image
    pixels_1x1_bgr = b'\x1e\x14\x0a' # BGR for (10, 20, 30)
    contents_1x1_rgb = np.array(create_bmp_24bpp(1, 1, pixels_1x1_bgr))

    list_of_inputs = []

    # --- Test cases for RGB content ---
    # Input 1: Decode RGB, default channels (0), should result in 3 channels
    list_of_inputs.append({
        'contents': contents_rgb,
        'channels': 0,
        'name': 'decode_rgb_default_channels'
    })

    # Input 2: Decode RGB, explicitly request 3 channels
    list_of_inputs.append({
        'contents': contents_rgb,
        'channels': 3,
        'name': 'decode_rgb_to_3_channels'
    })

    # Input 3: Decode RGB, convert to 4 channels (add alpha)
    list_of_inputs.append({
        'contents': contents_rgb,
        'channels': 4,
        'name': 'decode_rgb_to_4_channels'
    })

    # --- Test cases for RGBA content ---
    # Input 4: Decode RGBA, default channels (0), results in 3 channels (TF drops alpha by default)
    list_of_inputs.append({
        'contents': contents_rgba,
        'channels': 0,
        'name': 'decode_rgba_default_channels'
    })

    # Input 5: Decode RGBA, explicitly request 3 channels (drop alpha)
    list_of_inputs.append({
        'contents': contents_rgba,
        'channels': 3,
        'name': 'decode_rgba_to_3_channels'
    })

    # Input 6: Decode RGBA, explicitly request 4 channels
    list_of_inputs.append({
        'contents': contents_rgba,
        'channels': 4,
        'name': 'decode_rgba_to_4_channels'
    })
    
    # --- More test cases with different content and names ---
    # Input 7: Decode 1x1 RGB, default channels, no name
    list_of_inputs.append({
        'contents': contents_1x1_rgb,
        'channels': 0,
        'name': None
    })
    
    # Input 8: Decode 1x1 RGB, request 4 channels, no name
    list_of_inputs.append({
        'contents': contents_1x1_rgb,
        'channels': 4,
        'name': None
    })
    
    # Input 9: Decode RGBA, request 3 channels, empty name
    list_of_inputs.append({
        'contents': contents_rgba,
        'channels': 3,
        'name': ''
    })
    
    # Input 10: Decode RGB, default channels, empty name
    list_of_inputs.append({
        'contents': contents_rgb,
        'channels': 0,
        'name': ''
    })
    
    # Input 11: Another case for RGBA with 4 channels and no name
    list_of_inputs.append({
        'contents': contents_rgba,
        'channels': 4,
        'name': None
    })

    # Input 12: Another case for RGB with 3 channels and no name
    list_of_inputs.append({
        'contents': contents_rgb,
        'channels': 3,
        'name': None
    })

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
