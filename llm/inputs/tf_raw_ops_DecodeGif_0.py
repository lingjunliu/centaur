
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decode_gif_inputs():
    list_of_inputs = []

    # Input 1: Valid GIF-encoded image (empty)
    contents = np.array(b"").tobytes()
    name = None
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid GIF-encoded image (empty) with a name
    contents = np.array(b"").tobytes()
    name = "gif_decoder"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid GIF-encoded image (minimal valid GIF header)
    contents = np.array(b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;").tobytes()
    name = None
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Valid GIF-encoded image (minimal valid GIF header) with a name
    contents = np.array(b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;").tobytes()
    name = "another_gif"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Valid GIF-encoded image (slightly larger)
    contents = np.array(b"GIF89a\x02\x00\x02\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x02\x00\x02\x00\x00\x02\x02D\x01\x00;").tobytes()
    name = None
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Valid GIF-encoded image (slightly larger) with a name
    contents = np.array(b"GIF89a\x02\x00\x02\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x02\x00\x02\x00\x00\x02\x02D\x01\x00;").tobytes()
    name = "yet_another_gif"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Another minimal valid GIF
    contents = np.array(b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;').tobytes()
    name = None
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another minimal valid GIF with a name
    contents = np.array(b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;').tobytes()
    name = "gif_with_comment"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More complicated valid gif
    contents = np.array(b'GIF89a\x0A\x00\x0A\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x0A\x00\x0A\x00\x00\x02\x1f\x84\x8f\xa9\x93\x9b\xc9\x83Q\x8b\xa1\xbb`\xfb\x9b\xa3\x9c\xbb\x06\xcb\x8f\xad\xaf\xafY2\xd9\x05\r\x99+\xb7k\xad\xe3\x08\x00;').tobytes()
    name = None
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More complicated valid gif with name
    contents = np.array(b'GIF89a\x0A\x00\x0A\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x0A\x00\x0A\x00\x00\x02\x1f\x84\x8f\xa9\x93\x9b\xc9\x83Q\x8b\xa1\xbb`\xfb\x9b\xa3\x9c\xbb\x06\xcb\x8f\xad\xaf\xafY2\xd9\x05\r\x99+\xb7k\xad\xe3\x08\x00;').tobytes()
    name = "complex_gif"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DecodeGif"] = tf_raw_ops_decode_gif_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodeGif' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeGif'.")

check_valid('tf.raw_ops.DecodeGif', generated_inputs['tf.raw_ops.DecodeGif'], lib="tf", suffix=0)
