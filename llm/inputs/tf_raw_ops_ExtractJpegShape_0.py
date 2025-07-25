
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_extract_jpeg_shape_inputs():
    """
    This function generates a list of valid inputs for the
    tf.raw_ops.ExtractJpegShape operation without external dependencies.
    """
    # A valid 2x2 grayscale JPEG, generated via tf.image.encode_jpeg.
    _JPEG_GRAY_VALID = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00\x84\x00\t\x06\x07\x08\x07\x06\t\x08\x07\x08\n\n\t\x0b\r\x16\x0f\r\x0c\x0c\r\x1b\x14\x15\x10\x16\x1d\x1d\x1d\x1d\x1d\x1d\x1f\x1f\x1e\x1d\x1f\x1e\x1d\x1d\x1e\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\xff\xc0\x00\x11\x08\x00\x02\x00\x02\x01\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3Cr\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:DEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x0c\x01\x01\x00\x02\x11\x03\x11\x00?\x00\xf2\x9f\x03\x88\xff\xd9'
    # A valid 2x3 color JPEG, generated via tf.image.encode_jpeg.
    _JPEG_RGB_VALID = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00\x84\x00\t\x06\x07\x08\x07\x06\t\x08\x07\x08\n\n\t\x0b\r\x16\x0f\r\x0c\x0c\r\x1b\x14\x15\x10\x16\x1d\x1d\x1d\x1d\x1d\x1d\x1f\x1f\x1e\x1d\x1f\x1e\x1d\x1d\x1e\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x01\n\n\n\r\x0c\r\x1a\x0f\x0f\x1a\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\x1d\xff\xc0\x00\x11\x08\x00\x02\x00\x03\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3Cr\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:DEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11\x00\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00\x01\x02\x03\x11\x04\x05!1\x06\x12AQ\x07aq\x13"2\x81\x08\x14B\x91\xa1\xb1\xc1\t#3R\xf0\x15\xd1\n\x16$br\x82\t\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xd9\x83\x1f\xe5\x9f\n\xdc\x83/\xd6\xa2\xfe\x01/\xfa\x80\xff\xd9'

    list_of_inputs = []

    # Create a scalar numpy array of type object to hold the byte string.
    # This is to satisfy the test harness which expects numpy arrays for tensors
    # and has issues with fixed-length string dtypes ('S...').
    gray_contents = np.array(_JPEG_GRAY_VALID, dtype=object)
    rgb_contents = np.array(_JPEG_RGB_VALID, dtype=object)
    
    # Case 1: Grayscale image, default output_type (int32)
    input_dict = {
        'contents': gray_contents,
        'output_type': np.int32,
        'name': 'test_case_1_gray'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Grayscale image, int64 output_type, no name
    input_dict = {
        'contents': gray_contents,
        'output_type': np.int64,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Color image, default output_type (int32)
    input_dict = {
        'contents': rgb_contents,
        'output_type': np.int32,
        'name': 'test_case_3_rgb'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Color image, int64 output_type
    input_dict = {
        'contents': rgb_contents,
        'output_type': np.int64,
        'name': 'test_case_4_rgb_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Color image, no name
    input_dict = {
        'contents': rgb_contents,
        'output_type': np.int32,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Grayscale image, different name
    input_dict = {
        'contents': gray_contents,
        'output_type': np.int32,
        'name': 'another_gray_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Color image, int64 output, different name
    input_dict = {
        'contents': rgb_contents,
        'output_type': np.int64,
        'name': 'another_rgb_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Grayscale image, int64 output, with name
    input_dict = {
        'contents': gray_contents,
        'output_type': np.int64,
        'name': 'gray_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 9: Re-use grayscale with int32, no name
    input_dict = {
        'contents': gray_contents,
        'output_type': np.int32,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 10: Re-use color with int64, no name
    input_dict = {
        'contents': rgb_contents,
        'output_type': np.int64,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.ExtractJpegShape"] = tf_raw_ops_extract_jpeg_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ExtractJpegShape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ExtractJpegShape'.")

check_valid('tf.raw_ops.ExtractJpegShape', generated_inputs['tf.raw_ops.ExtractJpegShape'], lib="tf", suffix=0)
