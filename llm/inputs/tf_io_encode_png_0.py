
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_encode_png_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    compression = -1
    name = "test_image_1"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image = np.array([[[1, 2, 3, 4], [4, 5, 6, 7]], [[7, 8, 9, 10], [10, 11, 12, 13]]], dtype=np.uint8)
    compression = 0
    name = "test_image_2"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image = np.array([[[1], [4]], [[7], [10]]], dtype=np.uint8)
    compression = 5
    name = "test_image_3"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image = np.array([[[1, 2], [4, 5]], [[7, 8], [10, 11]]], dtype=np.uint8)
    compression = 9
    name = "test_image_4"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    compression = -1
    name = "test_image_5"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    image = np.array([[[1, 2, 3, 4], [4, 5, 6, 7]], [[7, 8, 9, 10], [10, 11, 12, 13]]], dtype=np.uint8)
    compression = 0
    name = "test_image_6"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    image = np.array([[[1], [4]], [[7], [10]]], dtype=np.uint8)
    compression = 5
    name = "test_image_7"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    image = np.array([[[1, 2], [4, 5]], [[7, 8], [10, 11]]], dtype=np.uint8)
    compression = 9
    name = "test_image_8"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - Higher dimension
    image = np.array([[[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]], dtype=np.uint8)
    compression = 7
    name = "test_image_9"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - 1 channel, uint8
    image = np.array([[[1], [4]], [[7], [10]]], dtype=np.uint8)
    compression = 2
    name = "test_image_10"
    input_dict = {"image": image, "compression": compression, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.encode_png"] = tf_io_encode_png_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.encode_png' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.encode_png'.")

check_valid('tf.io.encode_png', generated_inputs['tf.io.encode_png'], lib="tf", suffix=0)
