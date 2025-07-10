
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_flip_left_right_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D image
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 4D image (batch size 1)
    image = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.int32)
    seed = np.array([3, 4], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different image shape
    image = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.int32)
    seed = np.array([5, 6], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type (float32)
    image = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    seed = np.array([7, 8], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different seed values
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    seed = np.array([100, 200], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D image with batch size > 1
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.int32)
    seed = np.array([9, 10], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Grayscale image (channels = 1)
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)
    seed = np.array([11, 12], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger image
    image = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.int32)
    seed = np.array([13, 14], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.stateless_random_flip_left_right"] = tf_image_stateless_random_flip_left_right_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.stateless_random_flip_left_right' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_flip_left_right'.")

check_valid('tf.image.stateless_random_flip_left_right', generated_inputs['tf.image.stateless_random_flip_left_right'], lib="tf", suffix=0)
