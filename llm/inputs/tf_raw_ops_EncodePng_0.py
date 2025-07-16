
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_EncodePng_inputs():
    list_of_inputs = []

    # Input 1: Basic grayscale image
    image = np.zeros((10, 10, 1), dtype=np.uint8)
    compression = -1
    name = None
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: RGB image with compression level 0
    image = np.ones((20, 30, 3), dtype=np.uint8) * 255
    compression = 0
    name = "rgb_image"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: RGBA image with high compression
    image = np.random.randint(0, 256, size=(50, 40, 4), dtype=np.uint8)
    compression = 9
    name = "rgba_high_compression"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Grayscale image with alpha channel
    image = np.random.randint(0, 256, size=(15, 25, 2), dtype=np.uint8)
    compression = 5
    name = "grayscale_alpha"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint16 RGB image
    image = np.random.randint(0, 65536, size=(30, 20, 3), dtype=np.uint16)
    compression = -1
    name = "uint16_rgb"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint16 RGBA image
    image = np.random.randint(0, 65536, size=(40, 30, 4), dtype=np.uint16)
    compression = 3
    name = "uint16_rgba"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.EncodePng"] = []
for input_dict in tf_raw_ops_EncodePng_inputs():
    generated_inputs["tf.raw_ops.EncodePng"].append({
        "image": input_dict["image"],
        "compression": input_dict["compression"],
        "name": input_dict["name"]
    })

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EncodePng' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EncodePng'.")

check_valid('tf.raw_ops.EncodePng', generated_inputs['tf.raw_ops.EncodePng'], lib="tf", suffix=0)
