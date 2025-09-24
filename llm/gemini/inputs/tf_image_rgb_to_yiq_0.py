
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_rgb_to_yiq_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D image
    images = tf.constant(np.array([[0.0, 0.5, 1.0]], dtype=np.float32))
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D image
    images = tf.constant(np.array([[[0.2, 0.4, 0.6], [0.8, 0.1, 0.3]]], dtype=np.float32))
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple images
    images = tf.constant(np.array([[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]], dtype=np.float32))
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different values
    images = tf.constant(np.array([[0.9, 0.7, 0.5]], dtype=np.float32))
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array
    images = tf.constant(np.random.rand(2, 3, 4, 3).astype(np.float32))
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All zeros
    images = tf.constant(np.zeros((1, 1, 3), dtype=np.float32))
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All ones
    images = tf.constant(np.ones((1, 1, 3), dtype=np.float32))
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  Large image
    images = tf.constant(np.random.rand(100, 100, 3).astype(np.float32))
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Greyscale image
    images = tf.constant(np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32))
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another 3D image with more diverse values
    images = tf.constant(np.array([[[0.1, 0.8, 0.3], [0.9, 0.2, 0.7]]], dtype=np.float32))
    input_dict = {"images": images}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
temp_list = tf_image_rgb_to_yiq_inputs()
for i in range(len(temp_list)):
  generated_inputs["tf.image.rgb_to_yiq"] = [{"images":temp_list[i]["images"].numpy()}]
  break

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.rgb_to_yiq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.rgb_to_yiq'.")

check_valid('tf.image.rgb_to_yiq', generated_inputs['tf.image.rgb_to_yiq'], lib="tf", suffix=0)
