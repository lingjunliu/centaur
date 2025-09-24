
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_contrast_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    lower = 0.2
    upper = 0.5
    seed = np.array([1, 2], dtype=np.int32)
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image = np.random.rand(10, 10, 3).astype(np.float32)
    lower = 0.5
    upper = 1.5
    seed = np.array([123, 456], dtype=np.int32)
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image = np.random.rand(5, 5, 5, 3).astype(np.float32)
    lower = 0.1
    upper = 0.9
    seed = np.array([789, 101], dtype=np.int32)
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    lower = 0.7
    upper = 1.3
    seed = np.array([202, 303], dtype=np.int32)
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    image = np.random.rand(28, 28, 1).astype(np.float32)
    lower = 0.6
    upper = 1.4
    seed = np.array([404, 505], dtype=np.int32)
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    image = np.random.rand(32, 32, 3).astype(np.float32)
    lower = 0.8
    upper = 1.2
    seed = np.array([606, 707], dtype=np.int32)
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    image = np.ones((4, 4, 3), dtype=np.float32)
    lower = 0.9
    upper = 1.1
    seed = np.array([808, 909], dtype=np.int32)
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    image = np.zeros((4, 4, 3), dtype=np.float32)
    lower = 0.4
    upper = 0.6
    seed = np.array([1010, 1111], dtype=np.int32)
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    image = np.random.rand(1, 256, 256, 3).astype(np.float32)
    lower = 0.3
    upper = 0.7
    seed = np.array([1212, 1313], dtype=np.int32)
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    image = np.random.rand(64, 64, 3).astype(np.float32)
    lower = 0.25
    upper = 0.75
    seed = np.array([1414, 1515], dtype=np.int32)
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.stateless_random_contrast"] = tf_image_stateless_random_contrast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.stateless_random_contrast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_contrast'.")

check_valid('tf.image.stateless_random_contrast', generated_inputs['tf.image.stateless_random_contrast'], lib="tf", suffix=0)
