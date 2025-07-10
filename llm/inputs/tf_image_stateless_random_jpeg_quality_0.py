
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_jpeg_quality_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    min_jpeg_quality = 75
    max_jpeg_quality = 95
    seed = np.array([1, 2], dtype=np.int32)
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image = np.array([[[100, 150, 200], [50, 25, 0]]], dtype=np.uint8)
    min_jpeg_quality = 20
    max_jpeg_quality = 80
    seed = np.array([3, 4], dtype=np.int32)
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image = np.array([[[1, 2, 3]]], dtype=np.uint8)
    min_jpeg_quality = 0
    max_jpeg_quality = 100
    seed = np.array([5, 6], dtype=np.int32)
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image = np.array([[[10], [20]], [[30], [40]]], dtype=np.uint8)
    min_jpeg_quality = 50
    max_jpeg_quality = 60
    seed = np.array([7, 8], dtype=np.int32)
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    image = np.array([[[255, 0, 255]]], dtype=np.uint8)
    min_jpeg_quality = 1
    max_jpeg_quality = 99
    seed = np.array([9, 10], dtype=np.int32)
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    min_jpeg_quality = 0
    max_jpeg_quality = 1
    seed = np.array([11, 12], dtype=np.int32)
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    min_jpeg_quality = 99
    max_jpeg_quality = 100
    seed = np.array([13, 14], dtype=np.int32)
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    min_jpeg_quality = 50
    max_jpeg_quality = 51
    seed = np.array([15, 16], dtype=np.int32)
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8)
    min_jpeg_quality = 25
    max_jpeg_quality = 75
    seed = np.array([17, 18], dtype=np.int32)
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    image = np.array([[[50, 50, 50]]], dtype=np.uint8)
    min_jpeg_quality = 30
    max_jpeg_quality = 60
    seed = np.array([19, 20], dtype=np.int32)
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.stateless_random_jpeg_quality"] = tf_image_stateless_random_jpeg_quality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.stateless_random_jpeg_quality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_jpeg_quality'.")

check_valid('tf.image.stateless_random_jpeg_quality', generated_inputs['tf.image.stateless_random_jpeg_quality'], lib="tf", suffix=0)
