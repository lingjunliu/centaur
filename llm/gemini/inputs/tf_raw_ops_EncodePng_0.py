
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_EncodePng_inputs():
    list_of_inputs = []

    # Input 1: Grayscale (1 channel), uint8, default compression
    image_1 = np.random.randint(0, 256, size=(10, 10, 1), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_1,
        "compression": -1,
        "name": None
    })

    # Input 2: RGB (3 channels), uint8, highest compression (9)
    image_2 = np.random.randint(0, 256, size=(20, 20, 3), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_2,
        "compression": 9,
        "name": "encode_rgb_high"
    })

    # Input 3: RGBA (4 channels), uint8, no compression (0)
    image_3 = np.random.randint(0, 256, size=(15, 15, 4), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_3,
        "compression": 0,
        "name": "encode_rgba_8bit_no_comp"
    })

    # Input 4: Grayscale + Alpha (2 channels), uint8, medium compression (5)
    image_4 = np.random.randint(0, 256, size=(32, 32, 2), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_4,
        "compression": 5,
        "name": "gray_alpha"
    })

    # Input 5: Single pixel, RGB, uint8, default compression (-1)
    image_5 = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_5,
        "compression": -1,
        "name": "single_pixel_rgb"
    })

    # Input 6: Large image, Grayscale, uint8, fast compression (1)
    image_6 = np.random.randint(0, 256, size=(128, 128, 1), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_6,
        "compression": 1,
        "name": "large_gray_fast"
    })

    # Input 7: RGBA, uint8, compression level 4
    image_7 = np.random.randint(0, 256, size=(50, 50, 4), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_7,
        "compression": 4,
        "name": "rgba_mid"
    })

    # Input 8: RGB, uint8, compression level 7
    image_8 = np.random.randint(0, 256, size=(64, 48, 3), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_8,
        "compression": 7,
        "name": "rgb_8bit_high"
    })

    # Input 9: Grayscale + Alpha, uint8, compression level 3
    image_9 = np.random.randint(0, 256, size=(16, 32, 2), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_9,
        "compression": 3,
        "name": "gray_alpha_8bit"
    })

    # Input 10: Tall narrow image, RGB, uint8, compression level 8
    image_10 = np.random.randint(0, 256, size=(100, 10, 3), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_10,
        "compression": 8,
        "name": "tall_narrow_rgb"
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.EncodePng"] = tf_raw_ops_EncodePng_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.EncodePng' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EncodePng'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.EncodePng', generated_inputs['tf.raw_ops.EncodePng'], lib="tf", suffix=0)
