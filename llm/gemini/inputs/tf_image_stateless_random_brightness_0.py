
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_brightness_inputs():
    list_of_inputs = []

    # Input 1: Simple image, small max_delta
    image = np.array([[[0.5, 0.6, 0.7]]], dtype=np.float32)
    max_delta = 0.1
    seed = np.array([1, 2], dtype=np.int32)
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Larger image, larger max_delta
    image = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 0.0, 0.1]]], dtype=np.float32)
    max_delta = 0.3
    seed = np.array([3, 4], dtype=np.int32)
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Grayscale image
    image = np.array([[[0.5], [0.6]], [[0.7], [0.8]]], dtype=np.float32)
    max_delta = 0.2
    seed = np.array([5, 6], dtype=np.int32)
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Image with larger values
    image = np.array([[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]], dtype=np.float32)
    max_delta = 0.5
    seed = np.array([7, 8], dtype=np.int32)
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Image with values close to 0 and 1
    image = np.array([[[0.01, 0.99, 0.5]]], dtype=np.float32)
    max_delta = 0.05
    seed = np.array([9, 10], dtype=np.int32)
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different seed values
    image = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)
    max_delta = 0.15
    seed = np.array([11, 12], dtype=np.int32)
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multidimensional image
    image = np.random.rand(32, 32, 3).astype(np.float32)
    max_delta = 0.1
    seed = np.array([13, 14], dtype=np.int32)
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: max_delta close to 0
    image = np.array([[[0.3, 0.5, 0.7]]], dtype=np.float32)
    max_delta = 0.001
    seed = np.array([15, 16], dtype=np.int32)
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: max_delta close to 1
    image = np.array([[[0.3, 0.5, 0.7]]], dtype=np.float32)
    max_delta = 0.99
    seed = np.array([17, 18], dtype=np.int32)
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different int64 seed values
    image = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)
    max_delta = 0.15
    seed = np.array([2**31 - 1, 2**31 - 2], dtype=np.int32)
    input_dict = {"image": image, "max_delta": max_delta, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_brightness"] = tf_image_stateless_random_brightness_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.stateless_random_brightness' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_brightness'.")

check_valid('tf.image.stateless_random_brightness', generated_inputs['tf.image.stateless_random_brightness'], lib="tf", suffix=0)
