
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_EncodeJpegVariableQuality_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    images = np.zeros((100, 100, 3), dtype=np.uint8)
    quality = np.array(75, dtype=np.int32)
    name = None
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different image size
    images = np.zeros((50, 200, 3), dtype=np.uint8)
    quality = np.array(50, dtype=np.int32)
    name = "encode_jpeg_1"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: All white image
    images = np.ones((32, 32, 3), dtype=np.uint8) * 255
    quality = np.array(90, dtype=np.int32)
    name = "encode_jpeg_2"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Quality 0
    images = np.zeros((64, 64, 3), dtype=np.uint8)
    quality = np.array(0, dtype=np.int32)
    name = "encode_jpeg_3"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Quality 100
    images = np.zeros((128, 128, 3), dtype=np.uint8)
    quality = np.array(100, dtype=np.int32)
    name = "encode_jpeg_4"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small image
    images = np.zeros((1, 1, 3), dtype=np.uint8)
    quality = np.array(75, dtype=np.int32)
    name = "encode_jpeg_5"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Different number of channels (4)
    images = np.zeros((20, 20, 4), dtype=np.uint8)
    quality = np.array(80, dtype=np.int32)
    name = "encode_jpeg_6"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Greyscale image
    images = np.zeros((30, 30, 1), dtype=np.uint8)
    quality = np.array(60, dtype=np.int32)
    name = "encode_jpeg_7"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger image
    images = np.zeros((256, 256, 3), dtype=np.uint8)
    quality = np.array(25, dtype=np.int32)
    name = "encode_jpeg_8"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Non-square image
    images = np.zeros((64, 32, 3), dtype=np.uint8)
    quality = np.array(95, dtype=np.int32)
    name = "encode_jpeg_9"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.EncodeJpegVariableQuality"] = tf_raw_ops_EncodeJpegVariableQuality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EncodeJpegVariableQuality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EncodeJpegVariableQuality'.")

check_valid('tf.raw_ops.EncodeJpegVariableQuality', generated_inputs['tf.raw_ops.EncodeJpegVariableQuality'], lib="tf", suffix=0)
