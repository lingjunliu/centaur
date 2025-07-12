
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_image_inputs():
    list_of_inputs = []

    # Input 1: Minimal GIF (single frame)
    contents = np.array(b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;')
    channels = 0
    dtype = np.uint8
    name = "gif_decode_1"
    expand_animations = True

    input_dict = {
        "contents": contents,
        "channels": channels,
        "dtype": dtype,
        "name": name,
        "expand_animations": expand_animations
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: GIF single frame, expand_animations = False
    contents = np.array(b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;')
    channels = 0
    dtype = np.uint8
    name = "gif_decode_2"
    expand_animations = False

    input_dict = {
        "contents": contents,
        "channels": channels,
        "dtype": dtype,
        "name": name,
        "expand_animations": expand_animations
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: GIF Animated
    contents = np.array(b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x84\x01\x00;')
    channels = 0
    dtype = np.uint8
    name = "gif_decode_3"
    expand_animations = True

    input_dict = {
        "contents": contents,
        "channels": channels,
        "dtype": dtype,
        "name": name,
        "expand_animations": expand_animations
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: GIF Animated, expand_animations = False
    contents = np.array(b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x84\x01\x00;')
    channels = 0
    dtype = np.uint8
    name = "gif_decode_4"
    expand_animations = False

    input_dict = {
        "contents": contents,
        "channels": channels,
        "dtype": dtype,
        "name": name,
        "expand_animations": expand_animations
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_image"] = tf_io_decode_image_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.decode_image' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_image'.")

check_valid('tf.io.decode_image', generated_inputs['tf.io.decode_image'], lib="tf", suffix=0)
