
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_adjust_brightness_inputs():
    list_of_inputs = []

    # Input 1: Basic RGB image, positive delta
    image = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.0, 0.1, 0.2]]], dtype=np.float32)
    delta = 0.1
    input_dict = {"image": image, "delta": delta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic RGB image, negative delta
    image = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.0, 0.1, 0.2]]], dtype=np.float32)
    delta = -0.1
    input_dict = {"image": image, "delta": delta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Grayscale image, positive delta
    image = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    delta = 0.2
    input_dict = {"image": image, "delta": delta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Grayscale image, negative delta
    image = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    delta = -0.2
    input_dict = {"image": image, "delta": delta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D RGB image (batch of images), positive delta
    image = np.array([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.0, 0.1, 0.2]]],
                      [[[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], [[0.8, 0.9, 0.1], [0.1, 0.2, 0.3]]]], dtype=np.float32)
    delta = 0.05
    input_dict = {"image": image, "delta": delta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D RGB image (batch of images), negative delta
    image = np.array([[[[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], [[0.3, 0.2, 0.1], [0.2, 0.3, 0.4]]],
                      [[[0.8, 0.7, 0.6], [0.5, 0.4, 0.3]], [[0.2, 0.1, 0.9], [0.3, 0.4, 0.5]]]], dtype=np.float32)
    delta = -0.05
    input_dict = {"image": image, "delta": delta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D Grayscale image (batch of images), positive delta
    image = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    delta = 0.15
    input_dict = {"image": image, "delta": delta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D Grayscale image (batch of images), negative delta
    image = np.array([[[0.9, 0.8], [0.7, 0.6]], [[0.5, 0.4], [0.3, 0.2]]], dtype=np.float32)
    delta = -0.15
    input_dict = {"image": image, "delta": delta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large delta, positive
    image = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.0, 0.1, 0.2]]], dtype=np.float32)
    delta = 0.9
    input_dict = {"image": image, "delta": delta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large delta, negative
    image = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.0, 0.1, 0.2]]], dtype=np.float32)
    delta = -0.9
    input_dict = {"image": image, "delta": delta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D grayscale image
    image = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    delta = 0.1
    input_dict = {"image": image, "delta": delta}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: 5D RGB Image
    image = np.random.rand(2,2,2,2,3).astype(np.float32)
    delta = 0.2
    input_dict = {"image": image, "delta": delta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.adjust_brightness"] = tf_image_adjust_brightness_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.adjust_brightness' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_brightness'.")

check_valid('tf.image.adjust_brightness', generated_inputs['tf.image.adjust_brightness'], lib="tf", suffix=0)
