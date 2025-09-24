
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_saturation_inputs():
    list_of_inputs = []

    # Input 1, valid
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    lower = 0.5
    upper = 1.0
    seed = np.array([1, 2], dtype=np.int32)

    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    image = np.array([[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], [[1.0, 1.0, 1.0], [0.2, 0.4, 0.6]]], dtype=np.float32)
    lower = 1.0
    upper = 2.0
    seed = np.array([3, 4], dtype=np.int32)

    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    image = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)
    lower = 0.0
    upper = 0.5
    seed = np.array([5, 6], dtype=np.int32)

    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    image = np.array([[[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]], dtype=np.float32)
    lower = 0.75
    upper = 1.25
    seed = np.array([7, 8], dtype=np.int32)

    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5, valid, different shape
    image = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                     [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]], dtype=np.float32)
    lower = 0.2
    upper = 0.8
    seed = np.array([9, 10], dtype=np.int32)

    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    image = np.array([[[0.2, 0.4, 0.6], [0.8, 0.6, 0.4]]], dtype=np.float32)
    lower = 0.6
    upper = 1.4
    seed = np.array([11, 12], dtype=np.int32)
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, multi-channel image
    image = np.random.rand(32, 32, 3).astype(np.float32)
    lower = 0.3
    upper = 0.7
    seed = np.array([13, 14], dtype=np.int32)
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, different seed
    image = np.random.rand(16, 16, 3).astype(np.float32)
    lower = 0.8
    upper = 1.2
    seed = np.array([15, 16], dtype=np.int32)
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, smaller image
    image = np.random.rand(4, 4, 3).astype(np.float32)
    lower = 0.9
    upper = 1.1
    seed = np.array([17, 18], dtype=np.int32)
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid, grayscale-like image with some variation
    image = np.array([[[0.2, 0.21, 0.22], [0.4, 0.41, 0.42]], [[0.6, 0.61, 0.62], [0.8, 0.81, 0.82]]], dtype=np.float32)
    lower = 0.4
    upper = 0.6
    seed = np.array([19, 20], dtype=np.int32)

    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.stateless_random_saturation"] = tf_image_stateless_random_saturation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.stateless_random_saturation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_saturation'.")

check_valid('tf.image.stateless_random_saturation', generated_inputs['tf.image.stateless_random_saturation'], lib="tf", suffix=0)
