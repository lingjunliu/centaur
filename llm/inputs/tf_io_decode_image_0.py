
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_image_inputs():
    list_of_inputs = []

    # Input 1: Basic JPEG image
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\'(,(#!46/0-149;:\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x05\x11\x12\x06\x13\x14\x15\x16\x17\x18\x19\x1a!\x1b#\x1c$%\x1d\x1e\x1f\'()\x21\x22*\x23+\x24,\x25-\x26.\x27/\x2801\x292\x2a3\x2b4\x2c5\x2d6\x2e7\x2f8\x309\x31:\x32\x33\x34\x35\x36\x37\x38\x39\x3a\x3b\x3c\x3d\x3e\x3f\x40\x41\x42\x43\x44\x45\x46\x47\x48\x49\x4a\x4b\x4c\x4d\x4e\x4f\x50\x51\x52\x53\x54\x55\x56\x57\x58\x59\x5a\x5b\x5c\x5d\x5e\x5f\x60\x61\x62\x63\x64\x65\x66\x67\x68\x69\x6a\x6b\x6c\x6d\x6e\x6f\x70\x71\x72\x73\x74\x75\x76\x77\x78\x79\x7a\x7b\x7c\x7d\x7e\x7f\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf3\xfd\xff\xd9', dtype=np.bytes_)
    channels = np.int32(0)
    dtype = tf.uint8
    name = np.str_("decode_jpeg_1")
    expand_animations = np.bool_(True)

    input_dict = {
        "contents": contents,
        "channels": channels,
        "dtype": dtype,
        "name": name,
        "expand_animations": bool(expand_animations)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: PNG image
    contents = np.array(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\xfc\xff?\x03\x00\x05\xfa\x02\xfe\xa7\xcc\x00\x00\x00\x00IEND\xaeB`\x82', dtype=np.bytes_)
    channels = np.int32(3)
    dtype = tf.uint8
    name = np.str_("decode_png_1")
    expand_animations = np.bool_(False)

    input_dict = {
        "contents": contents,
        "channels": channels,
        "dtype": dtype,
        "name": name,
        "expand_animations": bool(expand_animations)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 3: GIF image
    contents = np.array(b'GIF89a\x01\x00\x01\x00\x00\xff\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;', dtype=np.bytes_)
    channels = np.int32(0)
    dtype = tf.uint8
    name = np.str_("decode_gif_1")
    expand_animations = np.bool_(True)

    input_dict = {
        "contents": contents,
        "channels": channels,
        "dtype": dtype,
        "name": name,
        "expand_animations": bool(expand_animations)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: BMP image
    contents = np.array(b'BM\x0e\x00\x00\x00\x00\x00\x00\x00\x0e\x00\x00\x00(\x00\x00\x00\x01\x00\x01\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.bytes_)
    channels = np.int32(0)
    dtype = tf.uint8
    name = np.str_("decode_bmp_1")
    expand_animations = np.bool_(True)

    input_dict = {
        "contents": contents,
        "channels": channels,
        "dtype": dtype,
        "name": name,
        "expand_animations": bool(expand_animations)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Remove invalid JPEG and replace with valid one
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x06\x04\x05\x06\x05\x04\x06\x06\x05\x06\x07\x07\x06\x08\n\x0e\n\x08\x08\n\x13\x0b\x0e\x11\x0b\x0b\x10\x16\x10\x10\x13\x1c\x14\x16\x12\x12\x17\x1f\x19\x1d\x1d\x1d\x1d\x15\x1e \x1c\x1c(%\x1f%\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01!\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x05\x11\x12\x06\x13\x14\x15\x16\x17\x18\x19\x1a!\x1b#\x1c$%\x1d\x1e\x1f\'()\x21\x22*\x23+\x24,\x25-\x26.\x27/\x2801\x292\x2a3\x2b4\x2c5\x2d6\x2e7\x2f8\x309\x31:\x32\x33\x34\x35\x36\x37\x38\x39\x3a\x3b\x3c\x3d\x3e\x3f\x40\x41\x42\x43\x44\x45\x46\x47\x48\x49\x4a\x4b\x4c\x4d\x4e\x4f\x50\x51\x52\x53\x54\x55\x56\x57\x58\x59\x5a\x5b\x5c\x5d\x5e\x5f\x60\x61\x62\x63\x64\x65\x66\x67\x68\x69\x6a\x6b\x6c\x6d\x6e\x6f\x70\x71\x72\x73\x74\x75\x76\x77\x78\x79\x7a\x7b\x7c\x7d\x7e\x7f\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\x85\xff\xd9', dtype=np.bytes_)
    channels = np.int32(1)
    dtype = tf.uint8
    name = np.str_("decode_jpeg_2")
    expand_animations = np.bool_(True)

    input_dict = {
        "contents": contents,
        "channels": channels,
        "dtype": dtype,
        "name": name,
        "expand_animations": bool(expand_animations)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different dtype
    contents = np.array(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\xfc\xff?\x03\x00\x05\xfa\x02\xfe\xa7\xcc\x00\x00\x00\x00IEND\xaeB`\x82', dtype=np.bytes_)
    channels = np.int32(0)
    dtype = tf.float32
    name = np.str_("decode_png_2")
    expand_animations = np.bool_(False)

    input_dict = {
        "contents": contents,
        "channels": channels,
        "dtype": dtype,
        "name": name,
        "expand_animations": bool(expand_animations)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty image
    contents = np.array(b'', dtype=np.bytes_)
    channels = np.int32(0)
    dtype = tf.uint8
    name = np.str_("decode_empty")
    expand_animations = np.bool_(True)

    input_dict = {
        "contents": contents,
        "channels": channels,
        "dtype": dtype,
        "name": name,
        "expand_animations": bool(expand_animations)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: channels = 4
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00', dtype=np.bytes_)
    channels = np.int32(4)
    dtype = tf.uint8
    name = np.str_("decode_jpeg_3")
    expand_animations = np.bool_(True)

    input_dict = {
        "contents": contents,
        "channels": channels,
        "dtype": dtype,
        "name": name,
        "expand_animations": bool(expand_animations)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different name
    contents = np.array(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\xfc\xff?\x03\x00\x05\xfa\x02\xfe\xa7\xcc\x00\x00\x00\x00IEND\xaeB`\x82', dtype=np.bytes_)
    channels = np.int32(0)
    dtype = tf.uint8
    name = np.str_("another_name")
    expand_animations = np.bool_(False)

    input_dict = {
        "contents": contents,
        "channels": channels,
        "dtype": dtype,
        "name": name,
        "expand_animations": bool(expand_animations)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Expand Animations True
    contents = np.array(b'GIF89a\x01\x00\x01\x00\x00\xff\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;', dtype=np.bytes_)
    channels = np.int32(0)
    dtype = tf.uint8
    name = np.str_("decode_gif_2")
    expand_animations = np.bool_(False)

    input_dict = {
        "contents": contents,
        "channels": channels,
        "dtype": dtype,
        "name": name,
        "expand_animations": bool(expand_animations)
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
