
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_adjust_gamma_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D image
    image = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    gamma = 0.5
    gain = 1.0
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D RGB image
    image = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 0.0, 0.5]]], dtype=np.float32)
    gamma = 2.0
    gain = 0.8
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Gamma < 1, gain > 1
    image = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    gamma = 0.8
    gain = 1.2
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Gamma > 1, gain < 1
    image = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    gamma = 1.5
    gain = 0.9
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different image shape
    image = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)
    gamma = 0.5
    gain = 1.0
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer Image
    image = np.array([[10, 50], [100, 200]], dtype=np.uint8)
    gamma = 0.5
    gain = 1.0
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D Image (Batch of Images)
    image = np.array([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 0.0, 0.5]]]], dtype=np.float32)
    gamma = 2.0
    gain = 0.8
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single Pixel Image
    image = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
    gamma = 1.0
    gain = 1.0
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Gamma close to zero
    image = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    gamma = 0.01
    gain = 1.0
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Gain close to zero
    image = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    gamma = 0.5
    gain = 0.01
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.adjust_gamma"] = tf_image_adjust_gamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.adjust_gamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_gamma'.")

check_valid('tf.image.adjust_gamma', generated_inputs['tf.image.adjust_gamma'], lib="tf", suffix=0)
