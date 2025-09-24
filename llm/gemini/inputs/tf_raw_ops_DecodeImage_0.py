
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_raw_ops_decodeimage_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.DecodeImage function.
    """
    list_of_inputs = []

    # --- Input 1: Basic JPEG, auto-detect channels, uint8 output ---
    jpeg_uint8_3ch = tf.image.encode_jpeg(
        np.random.randint(0, 256, (10, 8, 3), dtype=np.uint8)
    ).numpy()
    input_dict_1 = {
        'contents': np.array(jpeg_uint8_3ch, dtype=object),
        'channels': 0,
        'dtype': tf.uint8,
        'expand_animations': True,
        'name': 'jpeg_auto_channels_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # --- Input 2: Basic PNG, 3 channels specified, uint8 output ---
    png_uint8_3ch = tf.image.encode_png(
        np.random.randint(0, 256, (12, 12, 3), dtype=np.uint8)
    ).numpy()
    input_dict_2 = {
        'contents': np.array(png_uint8_3ch, dtype=object),
        'channels': 3,
        'dtype': tf.uint8,
        'expand_animations': True,
        'name': 'png_3_channels_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # --- Input 3: Decode JPEG to grayscale, float32 output ---
    input_dict_3 = {
        'contents': np.array(jpeg_uint8_3ch, dtype=object),
        'channels': 1,
        'dtype': tf.float32,
        'expand_animations': True,
        'name': 'jpeg_to_grayscale_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # --- Input 4: Decode uint16 PNG to uint16 output ---
    png_uint16_3ch = tf.image.encode_png(
        np.random.randint(0, 65536, (8, 8, 3), dtype=np.uint16)
    ).numpy()
    input_dict_4 = {
        'contents': np.array(png_uint16_3ch, dtype=object),
        'channels': 0,
        'dtype': tf.uint16,
        'expand_animations': True,
        'name': 'png_uint16_to_uint16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # --- Input 5: Disable animation expansion ---
    input_dict_5 = {
        'contents': np.array(jpeg_uint8_3ch, dtype=object),
        'channels': 3,
        'dtype': tf.uint8,
        'expand_animations': False,
        'name': 'no_animation_expansion'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # --- Input 6: Grayscale PNG to RGB ---
    png_uint8_1ch = tf.image.encode_png(
        np.random.randint(0, 256, (16, 16, 1), dtype=np.uint8)
    ).numpy()
    input_dict_6 = {
        'contents': np.array(png_uint8_1ch, dtype=object),
        'channels': 3,
        'dtype': tf.uint8,
        'expand_animations': True,
        'name': 'gray_png_to_rgb'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # --- Input 7: Grayscale JPEG (auto-detect channels) ---
    jpeg_uint8_1ch = tf.image.encode_jpeg(
        np.random.randint(0, 256, (20, 10, 1), dtype=np.uint8)
    ).numpy()
    input_dict_7 = {
        'contents': np.array(jpeg_uint8_1ch, dtype=object),
        'channels': 0,
        'dtype': tf.uint8,
        'expand_animations': True,
        'name': 'jpeg_grayscale_auto'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # --- Input 8: Different image size, float32 output, no animation ---
    jpeg_large = tf.image.encode_jpeg(
        np.random.randint(0, 256, (64, 32, 3), dtype=np.uint8)
    ).numpy()
    input_dict_8 = {
        'contents': np.array(jpeg_large, dtype=object),
        'channels': 3,
        'dtype': tf.float32,
        'expand_animations': False,
        'name': 'large_image_float_no_anim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # --- Input 9: PNG with alpha channel, auto-detect channels ---
    png_uint8_4ch = tf.image.encode_png(
        np.random.randint(0, 256, (5, 5, 4), dtype=np.uint8)
    ).numpy()
    input_dict_9 = {
        'contents': np.array(png_uint8_4ch, dtype=object),
        'channels': 0,
        'dtype': tf.uint8,
        'expand_animations': True,
        'name': 'png_with_alpha_auto'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # --- Input 10: PNG with alpha channel, force 3 channels (strip alpha) ---
    input_dict_10 = {
        'contents': np.array(png_uint8_4ch, dtype=object),
        'channels': 3,
        'dtype': tf.uint8,
        'expand_animations': True,
        'name': 'png_with_alpha_to_rgb'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # --- Input 11: PNG with alpha, force 1 channel (grayscale), uint16 output ---
    input_dict_11 = {
        'contents': np.array(png_uint8_4ch, dtype=object),
        'channels': 1,
        'dtype': tf.uint16,
        'expand_animations': False,
        'name': 'png_alpha_to_gray_uint16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.raw_ops.DecodeImage"] = get_tf_raw_ops_decodeimage_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DecodeImage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeImage'.")

check_valid('tf.raw_ops.DecodeImage', generated_inputs['tf.raw_ops.DecodeImage'], lib="tf", suffix=0)
