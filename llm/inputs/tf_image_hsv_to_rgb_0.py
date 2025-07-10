
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_hsv_to_rgb_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D array
    images = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    name = "simple_image"
    input_dict = {"images": tf.constant(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D array (batch of images)
    images = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.0, 0.1, 0.2]]], dtype=np.float32)
    name = "batch_image"
    input_dict = {"images": tf.constant(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single pixel
    images = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    name = "single_pixel"
    input_dict = {"images": tf.constant(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: All zeros
    images = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    name = "all_zeros"
    input_dict = {"images": tf.constant(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All ones
    images = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    name = "all_ones"
    input_dict = {"images": tf.constant(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different data type (bfloat16)
    images = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float16)
    name = "bfloat16_image"
    input_dict = {"images": tf.constant(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger image
    images = np.random.rand(100, 100, 3).astype(np.float32)
    name = "large_image"
    input_dict = {"images": tf.constant(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D array
    images = np.random.rand(2, 3, 4, 3).astype(np.float32)
    name = "4d_image"
    input_dict = {"images": tf.constant(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Half type
    images = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float16)
    name = "half_type"
    input_dict = {"images": tf.constant(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64
    images = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    name = "float64_type"
    input_dict = {"images": tf.constant(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.hsv_to_rgb"] = tf_image_hsv_to_rgb_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.hsv_to_rgb' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.hsv_to_rgb'.")

check_valid('tf.image.hsv_to_rgb', generated_inputs['tf.image.hsv_to_rgb'], lib="tf", suffix=0)
