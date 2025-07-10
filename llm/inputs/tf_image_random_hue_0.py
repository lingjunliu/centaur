
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_hue_inputs():
    list_of_inputs = []

    # Input 1
    image = np.random.rand(32, 32, 3).astype(np.float32)
    max_delta = 0.2
    seed = 123
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image = np.random.rand(1, 1, 3).astype(np.float32)
    max_delta = 0.0
    seed = 42
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image = np.random.rand(64, 64, 3).astype(np.float32)
    max_delta = 0.5
    seed = 0
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image = np.random.rand(128, 128, 3).astype(np.float32)
    max_delta = 0.1
    seed = -1
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    image = np.random.rand(256, 256, 3).astype(np.float32)
    max_delta = 0.3
    seed = 2**31 - 1  # Max int32
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    image = np.random.rand(4, 4, 3).astype(np.float32)
    max_delta = 0.4
    seed = -2**31  # Min int32
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    image = np.random.rand(10, 10, 3).astype(np.float32)
    max_delta = 0.05
    seed = 10000
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    image = np.random.rand(2, 2, 3).astype(np.float32)
    max_delta = 0.15
    seed = -10000
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    image = np.random.rand(3, 3, 3).astype(np.float32)
    max_delta = 0.25
    seed = 987654321
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    image = np.random.rand(5, 5, 3).astype(np.float32)
    max_delta = 0.35
    seed = -987654321
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.random_hue"] = tf_image_random_hue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.random_hue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_hue'.")

check_valid('tf.image.random_hue', generated_inputs['tf.image.random_hue'], lib="tf", suffix=0)
