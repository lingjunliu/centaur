
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_EncodePng_inputs():
    list_of_inputs = []

    # Input 1: Basic RGB image
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    compression = -1
    name = None
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Grayscale image
    image = np.zeros((10, 10, 1), dtype=np.uint8)
    compression = 0
    name = "grayscale"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: RGBA image
    image = np.zeros((10, 10, 4), dtype=np.uint8)
    compression = 9
    name = "rgba"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: uint16 grayscale
    image = np.zeros((10, 10, 1), dtype=np.uint16)
    compression = 5
    name = "uint16_grayscale"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint16 RGB
    image = np.zeros((10, 10, 3), dtype=np.uint16)
    compression = -1
    name = "uint16_rgb"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger image
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    compression = 2
    name = "large_image"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different image data
    image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    compression = 7
    name = "random_image"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another uint16 image
    image = np.random.randint(0, 65536, size=(10, 10, 3), dtype=np.uint16)
    compression = -1
    name = "random_uint16_image"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Grayscale + alpha
    image = np.zeros((10, 10, 2), dtype=np.uint8)
    compression = 3
    name = "grayscale_alpha"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Small image
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    compression = 1
    name = "small_image"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.EncodePng"] = tf_raw_ops_EncodePng_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EncodePng' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EncodePng'.")

check_valid('tf.raw_ops.EncodePng', generated_inputs['tf.raw_ops.EncodePng'], lib="tf", suffix=0)
