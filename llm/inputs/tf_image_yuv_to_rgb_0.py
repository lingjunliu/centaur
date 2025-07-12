
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_yuv_to_rgb_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D image
    images = np.array([[[0.5, 0.0, 0.0], [0.5, 0.2, 0.2]],
                       [[0.5, -0.2, -0.2], [0.5, 0.5, 0.5]]], dtype=np.float32)
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D image (batch of images)
    images = np.array([[[0.2, -0.3, 0.4], [0.8, 0.1, -0.5]],
                       [[0.9, 0.5, 0.0], [0.1, -0.4, 0.2]]], dtype=np.float64)
    images = np.stack([images, images], axis=0)
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different values
    images = np.array([[[1.0, 0.5, -0.5], [0.0, -0.5, 0.5]],
                       [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]], dtype=np.float32)
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shape
    images = np.random.rand(5, 10, 3).astype(np.float32)
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Another shape
    images = np.random.rand(1, 5, 5, 3).astype(np.float32)
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All zeros
    images = np.zeros((2, 2, 3), dtype=np.float32)
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All ones
    images = np.ones((2, 2, 3), dtype=np.float32)
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values within valid range
    images = np.array([[[0.1, -0.2, -0.3], [0.2, 0.0, -0.1]],
                       [[0.3, 0.4, 0.1], [0.4, 0.2, 0.3]]], dtype=np.float32)
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Values close to boundaries
    images = np.array([[[0.0, -0.5, 0.5], [1.0, 0.5, -0.5]],
                       [[0.5, -0.49, 0.49], [0.51, 0.48, -0.48]]], dtype=np.float32)
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More complex values
    images = np.array([[[0.7, -0.1, 0.3], [0.3, 0.2, -0.4]],
                       [[0.9, 0.3, -0.2], [0.1, -0.3, 0.4]]], dtype=np.float32)
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.yuv_to_rgb"] = tf_image_yuv_to_rgb_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.yuv_to_rgb' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.yuv_to_rgb'.")

check_valid('tf.image.yuv_to_rgb', generated_inputs['tf.image.yuv_to_rgb'], lib="tf", suffix=0)
