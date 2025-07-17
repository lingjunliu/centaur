
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_EncodePng_inputs():
    list_of_inputs = []

    # Input 1: Basic grayscale image
    image = np.zeros((100, 100, 1), dtype=np.uint8)
    input_dict = {"image": image, "compression": -1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic RGB image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    input_dict = {"image": image, "compression": 0, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: RGBA image with compression
    image = np.random.randint(0, 256, size=(50, 50, 4), dtype=np.uint8)
    input_dict = {"image": image, "compression": 5, "name": "my_png"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Grayscale + alpha
    image = np.ones((25, 25, 2), dtype=np.uint8) * 128
    input_dict = {"image": image, "compression": 9, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint16 grayscale image
    image = np.zeros((64, 64, 1), dtype=np.uint8)
    input_dict = {"image": image, "compression": -1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint16 RGB image
    image = np.random.randint(0, 2**8, size=(32, 32, 3), dtype=np.uint8)
    input_dict = {"image": image, "compression": 1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint16 RGBA image with compression
    image = np.ones((16, 16, 4), dtype=np.uint8) * 256
    input_dict = {"image": image, "compression": 7, "name": "test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small image
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    input_dict = {"image": image, "compression": -1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Grayscale image with non-zero values
    image = np.random.randint(0, 256, size=(40, 40, 1), dtype=np.uint8)
    input_dict = {"image": image, "compression": 2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Grayscale + alpha with non-zero values
    image = np.random.randint(0, 256, size=(30, 30, 2), dtype=np.uint8)
    input_dict = {"image": image, "compression": 3, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: uint16 grayscale image
    image = np.zeros((64, 64, 1), dtype=np.uint16)
    input_dict = {"image": image, "compression": -1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: uint16 RGB image
    image = np.random.randint(0, 2**8, size=(32, 32, 3), dtype=np.uint16)
    input_dict = {"image": image, "compression": 1, "name": None}
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
