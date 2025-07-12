
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_gif_inputs():
    list_of_inputs = []

    # Input 1: Simple valid GIF data (replace with actual valid gif bytes)
    gif_data = b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00\x21\xf9\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b'
    input_dict = {
        "contents": tf.constant(gif_data),
        "name": "gif_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Another valid GIF data with a different name
    gif_data = b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00\x21\xf9\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b'
    input_dict = {
        "contents": tf.constant(gif_data),
        "name": "gif_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Short name
    gif_data = b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00\x21\xf9\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b'
    input_dict = {
        "contents": tf.constant(gif_data),
        "name": "a"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Long name
    gif_data = b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00\x21\xf9\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b'
    input_dict = {
        "contents": tf.constant(gif_data),
        "name": "very_long_name_for_a_gif"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Valid GIF data as bytes
    gif_data = b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00\x21\xf9\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b'
    input_dict = {
        "contents": tf.constant(gif_data),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Name as empty string
    gif_data = b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00\x21\xf9\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b'
    input_dict = {
        "contents": tf.constant(gif_data),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Another valid gif
    gif_data = b'\x47\x49\x46\x38\x39\x61\x05\x00\x05\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff\x21\xf9\x04\x01\x00\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x05\x00\x05\x00\x00\x02\x02\x44\x01\x00\x3b'
    input_dict = {
        "contents": tf.constant(gif_data),
        "name": "valid_gif"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another valid gif, different name
    gif_data = b'\x47\x49\x46\x38\x39\x61\x05\x00\x05\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff\x21\xf9\x04\x01\x00\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x05\x00\x05\x00\x00\x02\x02\x44\x01\x00\x3b'
    input_dict = {
        "contents": tf.constant(gif_data),
        "name": "valid_gif_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Valid PNG data
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\xfc\xff\xff?\x03\x00\x06\xfc\x02\xfe\xa7\xcc\x00\x00\x00\x00IEND\xaeB`\x82'
    input_dict = {
        "contents": tf.constant(png_data),
        "name": "valid_png"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: empty name
    gif_data = b'\x47\x49\x46\x38\x39\x61\x05\x00\x05\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff\x21\xf9\x04\x01\x00\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x05\x00\x05\x00\x00\x02\x02\x44\x01\x00\x3b'
    input_dict = {
        "contents": tf.constant(gif_data),
        "name": ""
    }
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
