
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_extract_jpeg_shape_inputs():
    list_of_inputs = []

    # Input 1: Minimal valid input
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\xff\xd9', dtype='string')
    output_type = tf.int32
    name = None

    input_dict = {
        "contents": contents,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Changing output_type
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\xff\xd9', dtype='string')
    output_type = tf.int64
    name = None

    input_dict = {
        "contents": contents,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With a name
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\xff\xd9', dtype='string')
    output_type = tf.int32
    name = "extract_shape"

    input_dict = {
        "contents": contents,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different JPEG
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\xff\xd9', dtype='string')
    output_type = tf.int32
    name = None

    input_dict = {
        "contents": contents,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: Another valid jpeg data
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01!\x00\x02\x11\x01\xff\xd9', dtype='string')
    output_type = tf.int32
    name = None

    input_dict = {
        "contents": contents,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ExtractJpegShape"] = tf_raw_ops_extract_jpeg_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ExtractJpegShape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ExtractJpegShape'.")

check_valid('tf.raw_ops.ExtractJpegShape', generated_inputs['tf.raw_ops.ExtractJpegShape'], lib="tf", suffix=0)
