
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_io_decode_gif_inputs():
    list_of_inputs = []

    # Create a dummy GIF file for testing if it doesn't exist
    dummy_gif_path = "testdata/empty.gif"
    if not os.path.exists("testdata"):
        os.makedirs("testdata")
    if not os.path.exists(dummy_gif_path):
        with open(dummy_gif_path, "wb") as f:
            f.write(b"GIF89a\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;")

    dummy_txt_path = "testdata/empty.txt"
    if not os.path.exists(dummy_txt_path):
        with open(dummy_txt_path, "w") as f:
            f.write("Not a GIF")

    # Input 1: Valid GIF data (minimal)
    contents = tf.io.read_file(dummy_gif_path).numpy()
    name = None
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid GIF data with a name
    contents = tf.io.read_file(dummy_gif_path).numpy()
    name = "my_gif"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Another valid GIF with different dimensions
    contents = tf.io.read_file(dummy_gif_path).numpy()
    name = None
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Valid GIF with slightly invalid contents (but still a byte string)

    contents = b"Invalid GIF"
    name = None
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 5: A longer name
    contents = tf.io.read_file(dummy_gif_path).numpy()
    name = "a_very_long_name_for_a_gif"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    #Input 6 : Empty bytes
    contents = b""
    name = "empty"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    #Input 7 : GIF that does not start with GIF header
    contents = b"NOTGIF89a\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    name = None
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    #Input 8: GIF with a very short name
    contents = tf.io.read_file(dummy_gif_path).numpy()
    name = "a"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    #Input 9: Another GIF with a simple name
    contents = tf.io.read_file(dummy_gif_path).numpy()
    name = "test"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    #Input 10: Null name
    contents = tf.io.read_file(dummy_gif_path).numpy()
    name = ""
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_gif"] = tf_io_decode_gif_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.decode_gif' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_gif'.")

check_valid('tf.io.decode_gif', generated_inputs['tf.io.decode_gif'], lib="tf", suffix=0)
