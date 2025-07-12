
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_extract_jpeg_shape_inputs():
    list_of_inputs = []

    # Input 1 (Valid JPEG with more data)
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf6;\xff\xd9', dtype=np.string_)
    output_type = tf.int32
    name = "valid_jpeg"
    input_dict = {
        "contents": contents,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2 (Another valid JPEG)
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xc0\x00\x11\x08\x01\x00\x01\x00\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xa9\xff\xd9', dtype=np.string_)
    output_type = tf.int64
    name = "another_valid_jpeg"
    input_dict = {
        "contents": contents,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 (Valid JPEG, different dimensions)
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xc0\x00\x11\x08\x02\x00\x03\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xa9\xff\xd9', dtype=np.string_)
    output_type = tf.int32
    name = "diff_dims_jpeg"
    input_dict = {
        "contents": contents,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (Valid JPEG, another different dimensions)
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xc0\x00\x11\x08\x04\x00\x05\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xa9\xff\xd9', dtype=np.string_)
    output_type = tf.int64
    name = "another_diff_dims_jpeg"
    input_dict = {
        "contents": contents,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (Valid JPEG, larger dimensions)
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xc0\x00\x11\x08\x10\x00\x10\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xa9\xff\xd9', dtype=np.string_)
    output_type = tf.int32
    name = "larger_dims_jpeg"
    input_dict = {
        "contents": contents,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6 (Valid JPEG, even larger dimensions)
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xc0\x00\x11\x08\x80\x00\x80\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xa9\xff\xd9', dtype=np.string_)
    output_type = tf.int64
    name = "even_larger_dims_jpeg"
    input_dict = {
        "contents": contents,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7 (Valid JPEG, smaller dimensions)
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf6;\xff\xd9', dtype=np.string_)
    output_type = tf.int32
    name = "smaller_dims_jpeg"
    input_dict = {
        "contents": contents,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Minimal Valid JPEG (1x1) - Eliminating SOS/Huffman data
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x11\x00\xff\xd9', dtype=np.string_)
    output_type = tf.int32
    name = "minimal_valid_jpeg"
    input_dict = {
        "contents": contents,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.extract_jpeg_shape"] = tf_io_extract_jpeg_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.extract_jpeg_shape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.extract_jpeg_shape'.")

check_valid('tf.io.extract_jpeg_shape', generated_inputs['tf.io.extract_jpeg_shape'], lib="tf", suffix=0)
