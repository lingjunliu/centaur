
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_brightness_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D image
    image = np.array([[[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], [[0.2, 0.3, 0.4], [0.1, 0.2, 0.3]]], dtype=np.float32)
    max_delta = 0.2
    seed = 123
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Grayscale 2D image
    image = np.array([[0.5, 0.6], [0.8, 0.9]], dtype=np.float32)
    max_delta = 0.1
    seed = 456
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Image with values close to 0
    image = np.array([[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]], dtype=np.float32)
    max_delta = 0.05
    seed = 789
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Image with values close to 1
    image = np.array([[[0.97, 0.98, 0.99], [0.94, 0.95, 0.96]]], dtype=np.float32)
    max_delta = 0.03
    seed = 101
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D image (batch of images)
    image = np.array([[[[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], [[0.2, 0.3, 0.4], [0.1, 0.2, 0.3]]],
                      [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 0.1, 0.2]]]], dtype=np.float32)
    max_delta = 0.3
    seed = 234
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: max_delta close to 0
    image = np.array([[[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]]], dtype=np.float32)
    max_delta = 0.001
    seed = 345
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: max_delta close to 1
    image = np.array([[[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]], dtype=np.float32)
    max_delta = 0.99
    seed = 456
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: different seed
    image = np.array([[[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]]], dtype=np.float32)
    max_delta = 0.2
    seed = 567
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single pixel image
    image = np.array([[[0.5, 0.6, 0.7]]], dtype=np.float32)
    max_delta = 0.1
    seed = 678
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Image with a single channel
    image = np.array([[[0.5], [0.6]]], dtype=np.float32)
    max_delta = 0.3
    seed = 789
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.random_brightness"] = tf_image_random_brightness_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.random_brightness' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_brightness'.")

check_valid('tf.image.random_brightness', generated_inputs['tf.image.random_brightness'], lib="tf", suffix=0)
