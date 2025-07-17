
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_rgb_to_hsv_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D image
    images = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D image
    images = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]], dtype=np.float32)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D image
    images = np.random.rand(2, 3, 4, 3).astype(np.float32)
    input_dict = {"images": images, "name": "test_image"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half type
    images = np.array([[[0.5, 0.2, 0.8]]], dtype=np.float16)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 type
    images = np.array([[[0.7, 0.4, 0.6]]], dtype=np.float64)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: grayscale image (all channels equal)
    images = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: black image
    images = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: white image
    images = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D image (last dimension size 3)
    images = np.array([[0.2, 0.4, 0.6]], dtype=np.float32)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another 3D image with different values
    images = np.array([[[0.8, 0.3, 0.1], [0.2, 0.7, 0.9]], [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]], dtype=np.float32)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RGBToHSV"] = tf_raw_ops_rgb_to_hsv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RGBToHSV' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RGBToHSV'.")

check_valid('tf.raw_ops.RGBToHSV', generated_inputs['tf.raw_ops.RGBToHSV'], lib="tf", suffix=0)
