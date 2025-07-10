
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_jpeg_quality_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    min_jpeg_quality = 75
    max_jpeg_quality = 95
    seed = 123

    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image = np.array([[[1, 2, 3]]], dtype=np.uint8)
    min_jpeg_quality = 50
    max_jpeg_quality = 80
    seed = 456

    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8)
    min_jpeg_quality = 20
    max_jpeg_quality = 40
    seed = 789

    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    min_jpeg_quality = 0
    max_jpeg_quality = 100
    seed = 101

    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    image = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    min_jpeg_quality = 1
    max_jpeg_quality = 99
    seed = 202

    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    min_jpeg_quality = 30
    max_jpeg_quality = 70
    seed = 303

    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    image = np.ones((2, 2, 1), dtype=np.uint8) * 128
    min_jpeg_quality = 45
    max_jpeg_quality = 55
    seed = 404

    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    image = np.random.randint(0, 256, size=(3, 3, 3), dtype=np.uint8)
    min_jpeg_quality = 60
    max_jpeg_quality = 85
    seed = 505

    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    image = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    min_jpeg_quality = 5
    max_jpeg_quality = 15
    seed = 606

    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    image = np.array([[[1], [2]]], dtype=np.uint8)
    min_jpeg_quality = 88
    max_jpeg_quality = 98
    seed = 707

    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.random_jpeg_quality"] = tf_image_random_jpeg_quality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.random_jpeg_quality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_jpeg_quality'.")

check_valid('tf.image.random_jpeg_quality', generated_inputs['tf.image.random_jpeg_quality'], lib="tf", suffix=0)
