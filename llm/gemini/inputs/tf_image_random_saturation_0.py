
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_saturation_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    lower = 0.5
    upper = 1.5
    seed = 1
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    lower = 0.0
    upper = 1.0
    seed = 2
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)
    lower = 1.0
    upper = 2.0
    seed = 3
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image = np.array([[[0.8, 0.6, 0.4]]], dtype=np.float32)
    lower = 0.25
    upper = 0.75
    seed = 4
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    image = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
    lower = 0.7
    upper = 1.3
    seed = 5
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    image = np.random.rand(10, 10, 3).astype(np.float32)
    lower = 0.9
    upper = 1.1
    seed = 6
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    image = np.random.rand(5, 5, 3).astype(np.float32)
    lower = 0.0
    upper = 0.5
    seed = 7
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    image = np.random.rand(2, 3, 3).astype(np.float32)
    lower = 1.5
    upper = 2.5
    seed = 8
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    image = np.ones((4, 4, 3), dtype=np.float32)
    lower = 0.6
    upper = 0.8
    seed = 9
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    image = np.zeros((1, 1, 3), dtype=np.float32)
    lower = 1.2
    upper = 1.8
    seed = 10
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.random_saturation"] = tf_image_random_saturation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.random_saturation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_saturation'.")

check_valid('tf.image.random_saturation', generated_inputs['tf.image.random_saturation'], lib="tf", suffix=0)
