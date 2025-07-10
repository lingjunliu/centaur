
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_gif_inputs():
    list_of_inputs = []

    # Input 1: Empty GIF
    contents = tf.io.encode_jpeg(tf.constant([[[0, 0, 0]]], dtype=tf.uint8)).numpy().decode('utf-8').encode('utf-8')
    name = None
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Minimal GIF (invalid but tests the string type)
    contents = tf.io.encode_jpeg(tf.constant([[[0, 0, 0]]], dtype=tf.uint8)).numpy().decode('utf-8').encode('utf-8')
    name = "gif1"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Another minimal GIF with name
    contents = tf.io.encode_jpeg(tf.constant([[[0, 0, 0]]], dtype=tf.uint8)).numpy().decode('utf-8').encode('utf-8')
    name = "gif2"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: GIF with a slightly longer "name"
    contents = tf.io.encode_jpeg(tf.constant([[[0, 0, 0]]], dtype=tf.uint8)).numpy().decode('utf-8').encode('utf-8')
    name = "long_gif_name"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: GIF with special characters in the name
    contents = tf.io.encode_jpeg(tf.constant([[[0, 0, 0]]], dtype=tf.uint8)).numpy().decode('utf-8').encode('utf-8')
    name = "gif_with_chars#$% "
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: GIF with a unicode name
    contents = tf.io.encode_jpeg(tf.constant([[[0, 0, 0]]], dtype=tf.uint8)).numpy().decode('utf-8').encode('utf-8')
    name = "gif_unicode_name"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: GIF with numbers in the name
    contents = tf.io.encode_jpeg(tf.constant([[[0, 0, 0]]], dtype=tf.uint8)).numpy().decode('utf-8').encode('utf-8')
    name = "gif1234"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: GIF with longer content but still likely invalid
    contents = tf.io.encode_jpeg(tf.constant([[[0, 0, 0]]], dtype=tf.uint8)).numpy().decode('utf-8').encode('utf-8')
    name = None
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Short JPEG content with name
    contents = tf.io.encode_jpeg(tf.constant([[[0, 0, 0]]], dtype=tf.uint8)).numpy().decode('utf-8').encode('utf-8')
    name = "jpeg_name"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Short PNG content with name
    contents = tf.io.encode_png(tf.constant([[[0, 0, 0]]], dtype=tf.uint8)).numpy().decode('utf-8').encode('utf-8')
    name = "png_name"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: GIF with empty name
    contents = tf.io.encode_jpeg(tf.constant([[[0, 0, 0]]], dtype=tf.uint8)).numpy().decode('utf-8').encode('utf-8')
    name = ""
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Empty string
    contents = ""
    name = "empty_gif"
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
