
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_is_jpeg_inputs():
    list_of_inputs = []

    # Input 1: Empty string
    contents = np.array(b'')
    name = ""
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Minimal JPEG header
    contents = np.array(b'\xff\xd8\xff\xe0')
    name = "image1"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid JPEG header with a name
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF')
    name = "jpeg_header"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Longer JPEG-like content
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF' + b'some random bytes')
    name = None
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: JPEG start and end markers with other characters
    contents = np.array(b'\xff\xd8hello\xff\xd9')
    name = "test_image"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another JPEG header
    contents = np.array(b'\xff\xd8\xff\xe1\x00\x00Exif')
    name = "exif_image"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Content that is not a JPEG but has FF
    contents = np.array(b'\xffrandom stuff')
    name = "non_jpeg"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Slightly different JPEG header
    contents = np.array(b'\xff\xd8\xff\xe2\x00\x0bICC_PROFILE')
    name = "icc_profile"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: short jpeg like string
    contents = np.array(b'\xff\xd8\xff')
    name = "short_string"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: string with \xff\xd8 at end but not at the beginning
    contents = np.array(b'not jpeg\xff\xd8')
    name = "string_at_end"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.is_jpeg"] = tf_io_is_jpeg_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.is_jpeg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.is_jpeg'.")

check_valid('tf.io.is_jpeg', generated_inputs['tf.io.is_jpeg'], lib="tf", suffix=0)
