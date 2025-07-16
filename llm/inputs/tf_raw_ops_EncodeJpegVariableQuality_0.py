
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
    name = "jpeg_encode_1"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different image size
    images = np.zeros((50, 200, 3), dtype=np.uint8)
    quality = np.array(50, dtype=np.int32)
    name = "jpeg_encode_2"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: All white image
    images = np.full((32, 32, 3), 255, dtype=np.uint8)
    quality = np.array(90, dtype=np.int32)
    name = "jpeg_encode_3"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Grayscale image
    images = np.zeros((64, 64, 1), dtype=np.uint8)
    images = np.concatenate([images, images, images], axis=2) #Make it RGB
    quality = np.array(25, dtype=np.int32)
    name = "jpeg_encode_4"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Minimum quality
    images = np.random.randint(0, 256, size=(28, 28, 3), dtype=np.uint8)
    quality = np.array(0, dtype=np.int32)
    name = "jpeg_encode_5"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Maximum quality
    images = np.random.randint(0, 256, size=(28, 28, 3), dtype=np.uint8)
    quality = np.array(100, dtype=np.int32)
    name = "jpeg_encode_6"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Square image
    images = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
    quality = np.array(60, dtype=np.int32)
    name = "jpeg_encode_7"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small image
    images = np.random.randint(0, 256, size=(16, 16, 3), dtype=np.uint8)
    quality = np.array(80, dtype=np.int32)
    name = "jpeg_encode_8"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different image with some pattern
    images = np.tile(np.array([[[0, 0, 0], [255, 255, 255]], [[255, 255, 255], [0, 0, 0]]], dtype=np.uint8), (16, 16, 1))
    quality = np.array(40, dtype=np.int32)
    name = "jpeg_encode_9"
    input_dict = {"images": images, "quality": quality, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another size
    images = np.random.randint(0, 256, size=(256, 128, 3), dtype=np.uint8)
    quality = np.array(70, dtype=np.int32)
    name = "jpeg_encode_10"
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
