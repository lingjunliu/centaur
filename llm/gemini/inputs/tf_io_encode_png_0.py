
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_encode_png_inputs():
    list_of_inputs = []
    
    # Input 1: Grayscale (1 channel), uint8, default compression
    input_dict = {
        'image': np.zeros((10, 10, 1), dtype=np.uint8),
        'compression': -1,
        'name': "gray_img"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Grayscale + Alpha (2 channels), uint8, compression 0
    input_dict = {
        'image': np.ones((8, 8, 2), dtype=np.uint8) * 128,
        'compression': 0,
        'name': "gray_alpha_img"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: RGB (3 channels), uint8, max compression
    input_dict = {
        'image': np.random.randint(0, 256, size=(12, 12, 3), dtype=np.uint8),
        'compression': 9,
        'name': "rgb_img"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: RGBA (4 channels), uint8, compression 5
    input_dict = {
        'image': np.zeros((16, 16, 4), dtype=np.uint8),
        'compression': 5,
        'name': "rgba_img"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batch of Grayscale, uint8, compression 1
    input_dict = {
        'image': np.ones((2, 10, 10, 1), dtype=np.uint8) * 200,
        'compression': 1,
        'name': "batch_gray"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batch of RGB, uint8, default compression
    input_dict = {
        'image': np.zeros((1, 15, 15, 3), dtype=np.uint8),
        'compression': -1,
        'name': "batch_rgb"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: RGB, uint8, compression 6
    input_dict = {
        'image': np.random.randint(0, 256, size=(20, 20, 3), dtype=np.uint8),
        'compression': 6,
        'name': "rgb_uint8_alternative"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Grayscale, uint8, compression 3
    input_dict = {
        'image': np.ones((5, 5, 1), dtype=np.uint8) * 128,
        'compression': 3,
        'name': "gray_uint8_alternative"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: RGBA, uint8, compression 8
    input_dict = {
        'image': np.zeros((2, 2, 4), dtype=np.uint8),
        'compression': 8,
        'name': "rgba_uint8_alternative"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Rank 5 Batch of RGB, uint8, compression 2
    input_dict = {
        'image': np.ones((1, 2, 4, 4, 3), dtype=np.uint8) * 100,
        'compression': 2,
        'name': "rank5_rgb"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.io.encode_png"] = tf_io_encode_png_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.encode_png' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.encode_png'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.io.encode_png', generated_inputs['tf.io.encode_png'], lib="tf", suffix=0)
