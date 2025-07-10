
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_adjust_contrast_inputs():
    list_of_inputs = []

    # Input 1: Simple 3D array, positive contrast factor
    images = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    contrast_factor = 2.0
    input_dict = {"images": images, "contrast_factor": np.float32(contrast_factor)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple 3D array, negative contrast factor
    images = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    contrast_factor = -0.5
    input_dict = {"images": images, "contrast_factor": np.float32(contrast_factor)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Simple 3D array, zero contrast factor
    images = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    contrast_factor = 0.0
    input_dict = {"images": images, "contrast_factor": np.float32(contrast_factor)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Simple 3D array, contrast factor = 1.0
    images = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    contrast_factor = 1.0
    input_dict = {"images": images, "contrast_factor": np.float32(contrast_factor)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array (batch of images), positive contrast
    images = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
                       [[[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]]], dtype=np.float32)
    contrast_factor = 1.5
    input_dict = {"images": images, "contrast_factor": np.float32(contrast_factor)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array (batch of images), negative contrast
    images = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
                       [[[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]]], dtype=np.float32)
    contrast_factor = -0.8
    input_dict = {"images": images, "contrast_factor": np.float32(contrast_factor)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Grayscale image (3D with channels=1), positive contrast
    images = np.array([[[0.1], [0.2]], [[0.3], [0.4]]], dtype=np.float32)
    contrast_factor = 3.0
    input_dict = {"images": images, "contrast_factor": np.float32(contrast_factor)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Grayscale image (3D with channels=1), negative contrast
    images = np.array([[[0.1], [0.2]], [[0.3], [0.4]]], dtype=np.float32)
    contrast_factor = -0.2
    input_dict = {"images": images, "contrast_factor": np.float32(contrast_factor)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger image size, positive contrast
    images = np.random.rand(100, 100, 3).astype(np.float32)
    contrast_factor = 0.7
    input_dict = {"images": images, "contrast_factor": np.float32(contrast_factor)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different dtype (float64), positive contrast
    images = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float64)
    contrast_factor = 2.0
    input_dict = {"images": images, "contrast_factor": np.float32(contrast_factor)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.adjust_contrast"] = tf_image_adjust_contrast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.adjust_contrast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_contrast'.")

check_valid('tf.image.adjust_contrast', generated_inputs['tf.image.adjust_contrast'], lib="tf", suffix=0)
