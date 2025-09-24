
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_grayscale_to_rgb_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D grayscale image
    images = np.array([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float32)
    name = "grayscale_to_rgb_1"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D grayscale image (batch of images)
    images = np.array([[[1.0]], [[2.0]]], dtype=np.float32)
    images = np.stack([images, images + 1], axis=0)
    name = "grayscale_to_rgb_2"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different dtype (float64)
    images = np.array([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float64)
    name = "grayscale_to_rgb_3"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Grayscale with negative values
    images = np.array([[[ -1.0]], [[0.0]], [[1.0]]], dtype=np.float32)
    name = "grayscale_to_rgb_4"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Grayscale with larger values
    images = np.array([[[100.0]], [[200.0]], [[255.0]]], dtype=np.float32)
    name = "grayscale_to_rgb_5"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shape
    images = np.array([[[0.1]], [[0.2]], [[0.3]], [[0.4]]], dtype=np.float32)
    name = "grayscale_to_rgb_6"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  Valid batch with same shape
    images1 = np.array([[[0.1]], [[0.2]], [[0.3]]], dtype=np.float32)
    images2 = np.array([[[0.4]], [[0.5]], [[0.6]]], dtype=np.float32)
    images = np.stack([images1, images2], axis=0)
    name = "grayscale_to_rgb_7"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty name
    images = np.array([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float32)
    name = ""
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another batch
    images = np.array([[[0.5]], [[0.6]], [[0.7]]], dtype=np.float32)
    images = np.stack([images, images * 2], axis=0)
    name = "grayscale_to_rgb_9"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More complex image data
    images = np.array([[[0.1]], [[0.5]], [[0.9]]], dtype=np.float32)
    images = np.stack([images, images - 0.2], axis=0)
    name = "grayscale_to_rgb_10"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.grayscale_to_rgb"] = tf_image_grayscale_to_rgb_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.grayscale_to_rgb' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.grayscale_to_rgb'.")

check_valid('tf.image.grayscale_to_rgb', generated_inputs['tf.image.grayscale_to_rgb'], lib="tf", suffix=0)
