
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_hue_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    max_delta = 0.2
    seed = np.array([1, 2], dtype=np.int32)

    input_dict = {
        "image": tf.convert_to_tensor(image),
        "max_delta": max_delta,
        "seed": tf.convert_to_tensor(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
    max_delta = 0.0
    seed = np.array([0, 0], dtype=np.int32)

    input_dict = {
        "image": tf.convert_to_tensor(image),
        "max_delta": max_delta,
        "seed": tf.convert_to_tensor(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
    max_delta = 0.5
    seed = np.array([123, 456], dtype=np.int32)

    input_dict = {
        "image": tf.convert_to_tensor(image),
        "max_delta": max_delta,
        "seed": tf.convert_to_tensor(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image = np.array([[[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]]], dtype=np.float32)
    max_delta = 0.1
    seed = np.array([7, 8], dtype=np.int32)

    input_dict = {
        "image": tf.convert_to_tensor(image),
        "max_delta": max_delta,
        "seed": tf.convert_to_tensor(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    image = np.array([[[0.9, 0.7, 0.5]]], dtype=np.float32)
    max_delta = 0.3
    seed = np.array([9, 10], dtype=np.int32)

    input_dict = {
        "image": tf.convert_to_tensor(image),
        "max_delta": max_delta,
        "seed": tf.convert_to_tensor(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    image = np.array([[[0.1, 0.3, 0.5]]], dtype=np.float32)
    max_delta = 0.4
    seed = np.array([11, 12], dtype=np.int32)

    input_dict = {
        "image": tf.convert_to_tensor(image),
        "max_delta": max_delta,
        "seed": tf.convert_to_tensor(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    image = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    max_delta = 0.25
    seed = np.array([13, 14], dtype=np.int32)

    input_dict = {
        "image": tf.convert_to_tensor(image),
        "max_delta": max_delta,
        "seed": tf.convert_to_tensor(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
   image = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
    max_delta = 0.15
    seed = np.array([15, 16], dtype=np.int32)

    input_dict = {
        "image": tf.convert_to_tensor(image),
        "max_delta": max_delta,
        "seed": tf.convert_to_tensor(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    image = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
    max_delta = 0.35
    seed = np.array([17, 18], dtype=np.int32)

    input_dict = {
        "image": tf.convert_to_tensor(image),
        "max_delta": max_delta,
        "seed": tf.convert_to_tensor(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    image = np.array([[[0.3, 0.6, 0.9]]], dtype=np.float32)
    max_delta = 0.05
    seed = np.array([19, 20], dtype=np.int32)

    input_dict = {
        "image": tf.convert_to_tensor(image),
        "max_delta": max_delta,
        "seed": tf.convert_to_tensor(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.stateless_random_hue"] = tf_image_stateless_random_hue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.stateless_random_hue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_hue'.")

check_valid('tf.image.stateless_random_hue', generated_inputs['tf.image.stateless_random_hue'], lib="tf", suffix=0)
