
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_grayscale_to_rgb_inputs():
    list_of_inputs = []

    # Input 1: Basic grayscale image
    images = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    name = "basic_grayscale"
    input_dict = {"images": tf.constant(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Grayscale image with multiple channels (should still work)
    images = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    name = "multiple_channels"
    input_dict = {"images": tf.constant(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Grayscale image with negative values
    images = np.array([[[ -1.0], [0.0], [1.0]]], dtype=np.float32)
    name = "negative_values"
    input_dict = {"images": tf.constant(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Grayscale image with different data type
    images = np.array([[[1], [2], [3]]], dtype=np.int32)
    name = "integer_values"
    input_dict = {"images": tf.cast(tf.constant(images), tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Grayscale image with larger values
    images = np.array([[[100.0], [200.0], [255.0]]], dtype=np.float32)
    name = "larger_values"
    input_dict = {"images": tf.constant(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor
    images = np.random.rand(2, 3, 4, 1).astype(np.float32)
    name = "4d_tensor"
    input_dict = {"images": tf.constant(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor with multiple images
    images = np.array([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]], dtype=np.float32)
    name = "multiple_images"
    input_dict = {"images": tf.constant(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D tensor
    images = np.random.rand(1, 2, 3, 4, 1).astype(np.float32)
    name = "5d_tensor"
    input_dict = {"images": tf.constant(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Empty name
    images = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    name = ""
    input_dict = {"images": tf.constant(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different shape
    images = np.array([[[0.5], [0.6]], [[0.7], [0.8]]], dtype=np.float32)
    name = "different_shape"
    input_dict = {"images": tf.constant(images), "name": name}
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
