
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_encode_jpeg_variable_quality_inputs():
    list_of_inputs = []

    # Input 1
    images = np.uint8(np.random.randint(0, 256, size=(64, 64, 3)))
    quality = np.int32(75)
    name = "jpeg_encode_1"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    images = np.uint8(np.random.randint(0, 256, size=(128, 128, 3)))
    quality = np.int32(90)
    name = "jpeg_encode_2"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    images = np.uint8(np.random.randint(0, 256, size=(32, 32, 3)))
    quality = np.int32(50)
    name = "jpeg_encode_3"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    images = np.uint8(np.random.randint(0, 256, size=(256, 256, 3)))
    quality = np.int32(25)
    name = "jpeg_encode_4"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    images = np.uint8(np.random.randint(0, 256, size=(100, 150, 3)))
    quality = np.int32(100)
    name = "jpeg_encode_5"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    images = np.uint8(np.random.randint(0, 256, size=(200, 100, 3)))
    quality = np.int32(0)
    name = "jpeg_encode_6"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    images = np.uint8(np.random.randint(0, 256, size=(50, 50, 3)))
    quality = np.int32(1)
    name = "jpeg_encode_7"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    images = np.uint8(np.random.randint(0, 256, size=(120, 80, 3)))
    quality = np.int32(99)
    name = "jpeg_encode_8"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    images = np.uint8(np.random.randint(0, 256, size=(80, 120, 3)))
    quality = np.int32(55)
    name = "jpeg_encode_9"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    images = np.uint8(np.random.randint(0, 256, size=(40, 40, 3)))
    quality = np.int32(45)
    name = "jpeg_encode_10"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.EncodeJpegVariableQuality"] = tf_raw_ops_encode_jpeg_variable_quality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EncodeJpegVariableQuality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EncodeJpegVariableQuality'.")

check_valid('tf.raw_ops.EncodeJpegVariableQuality', generated_inputs['tf.raw_ops.EncodeJpegVariableQuality'], lib="tf", suffix=0)
