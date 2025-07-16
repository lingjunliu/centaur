
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decodeimage_inputs():
    list_of_inputs = []

    # Input 1: Minimal valid input (JPEG image)
    contents = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00...\xff\xd9'
    channels = 0
    dtype = tf.uint8
    expand_animations = True
    name = "decode_image_1"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "expand_animations": expand_animations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Specify channels (JPEG image)
    contents = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00...\xff\xd9'
    channels = 3
    dtype = tf.uint8
    expand_animations = True
    name = "decode_image_2"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "expand_animations": expand_animations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Use uint16 dtype (PNG image)
    contents = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\xda\xed\xc1\x01\x01\x00\x00\x00\xc2\xa0\xf7Om\x00\x00\x00\x00IEND\xaeB`\x82'
    channels = 0
    dtype = tf.uint16
    expand_animations = True
    name = "decode_image_3"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "expand_animations": expand_animations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Expand Animations False (GIF image, first frame only)
    contents = b'GIF89a\x01\x00\x01\x00\x00\xff\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
    channels = 0
    dtype = tf.uint8
    expand_animations = False
    name = "decode_image_4"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "expand_animations": expand_animations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: PNG image
    contents = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\xda\xed\xc1\x01\x01\x00\x00\x00\xc2\xa0\xf7Om\x00\x00\x00\x00IEND\xaeB`\x82'
    channels = 4
    dtype = tf.uint8
    expand_animations = True
    name = "decode_image_5"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "expand_animations": expand_animations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: BMP image
    contents = b'BM\x0e\x00\x00\x00\x00\x00\x00\x00\x0e\x00\x00\x00(\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff'
    channels = 0
    dtype = tf.uint8
    expand_animations = True
    name = "decode_image_6"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "expand_animations": expand_animations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Specify channels for BMP
    contents = b'BM\x0e\x00\x00\x00\x00\x00\x00\x00\x0e\x00\x00\x00(\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff'
    channels = 1
    dtype = tf.uint8
    expand_animations = True
    name = "decode_image_7"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "expand_animations": expand_animations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float32 dtype (PNG image)
    contents = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\xda\xed\xc1\x01\x01\x00\x00\x00\xc2\xa0\xf7Om\x00\x00\x00\x00IEND\xaeB`\x82'
    channels = 0
    dtype = tf.float32
    expand_animations = True
    name = "decode_image_8"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "expand_animations": expand_animations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: Simple GIF, animation expanded
    contents = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
    channels = 0
    dtype = tf.uint8
    expand_animations = True
    name = "decode_image_9"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "expand_animations": expand_animations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different name
    contents = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00...\xff\xd9'
    channels = 0
    dtype = tf.uint8
    expand_animations = True
    name = "another_name"
    input_dict = {"contents": contents, "channels": channels, "dtype": dtype, "expand_animations": expand_animations, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DecodeImage"] = tf_raw_ops_decodeimage_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodeImage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeImage'.")

check_valid('tf.raw_ops.DecodeImage', generated_inputs['tf.raw_ops.DecodeImage'], lib="tf", suffix=0)
