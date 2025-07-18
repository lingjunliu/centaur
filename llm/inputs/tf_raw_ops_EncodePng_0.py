
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import torch

def tf_raw_ops_encode_png_inputs():
    list_of_inputs = []

    # Input 1: Basic uint8 RGB image with default compression
    input_dict_1 = {
        'image': np.random.randint(0, 256, size=(10, 20, 3), dtype=np.uint8),
        'compression': -1,
        'name': 'rgb_default_compression'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: uint8 grayscale image with no compression
    input_dict_2 = {
        'image': np.random.randint(0, 256, size=(32, 32, 1), dtype=np.uint8),
        'compression': 0,
        'name': 'grayscale_no_compression'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: uint8 RGBA image with maximum compression
    input_dict_3 = {
        'image': np.random.randint(0, 256, size=(16, 16, 4), dtype=np.uint8),
        'compression': 9,
        'name': 'rgba_max_compression'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: uint8 grayscale + alpha image with medium compression
    input_dict_4 = {
        'image': np.random.randint(0, 256, size=(25, 15, 2), dtype=np.uint8),
        'compression': 5,
        'name': 'grayscale_alpha_medium_compression'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Large dimensions uint8 image
    input_dict_5 = {
        'image': np.random.randint(0, 256, size=(100, 120, 3), dtype=np.uint8),
        'compression': 2,
        'name': 'large_image_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Minimal dimensions (1x1) uint8 image
    input_dict_6 = {
        'image': np.array([[[128]]], dtype=np.uint8),
        'compression': 7,
        'name': 'minimal_image_1x1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: uint8 image with a non-square aspect ratio
    input_dict_7 = {
        'image': np.random.randint(0, 256, size=(50, 10, 4), dtype=np.uint8),
        'compression': 3,
        'name': 'tall_image_rgba'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Empty string for name
    input_dict_8 = {
        'image': np.random.randint(0, 256, size=(5, 5, 3), dtype=np.uint8),
        'compression': 6,
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Wide uint8 grayscale image
    input_dict_9 = {
        'image': np.random.randint(0, 256, size=(10, 100, 1), dtype=np.uint8),
        'compression': 1,
        'name': 'wide_grayscale_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Square uint8 grayscale+alpha image
    input_dict_10 = {
        'image': np.random.randint(0, 256, size=(64, 64, 2), dtype=np.uint8),
        'compression': 8,
        'name': 'square_ga_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.EncodePng"] = tf_raw_ops_encode_png_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.EncodePng' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EncodePng'.")

check_valid('tf.raw_ops.EncodePng', generated_inputs['tf.raw_ops.EncodePng'], lib="tf", suffix=0)
