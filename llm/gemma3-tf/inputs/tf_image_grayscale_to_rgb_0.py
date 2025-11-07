
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_grayscale_to_rgb_inputs():
    list_of_inputs = []

    images1 = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    name1 = "example_1"
    input_dict1 = {"images": images1, "name": name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    images2 = np.array([[[[0.5]], [[0.8]]]], dtype=np.float32)
    name2 = "example_2"
    input_dict2 = {"images": images2, "name": name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    images3 = np.array([[[[-1.0], [0.0]]]], dtype=np.float32)
    name3 = "example_3"
    input_dict3 = {"images": images3, "name": name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    images4 = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    name4 = "example_4"
    input_dict4 = {"images": images4, "name": name4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    images5 = np.array([[[[0.1], [0.2]], [[0.3], [0.4]]], [[ [0.5], [0.6]], [[0.7], [0.8]]]], dtype=np.float32)
    name5 = "example_5"
    input_dict5 = {"images": images5, "name": name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    images6 = np.array([[[[1.0]]]], dtype=np.float64)
    name6 = "example_6"
    input_dict6 = {"images": images6, "name": name6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    images7 = np.array([[[[255.0]]]], dtype=np.uint8)
    name7 = "example_7"
    input_dict7 = {"images": images7, "name": name7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    images8 = np.array([[[[0.0]]]], dtype=np.float16)
    name8 = "example_8"
    input_dict8 = {"images": images8, "name": name8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    images9 = np.array([[[[1.0], [-1.0]]]], dtype=np.float32)
    name9 = "example_9"
    input_dict9 = {"images": images9, "name": name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    images10 = np.array([[[[0.5]]]], dtype=np.complex64)
    name10 = "example_10"
    input_dict10 = {"images": images10, "name": name10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.grayscale_to_rgb"] = tf_image_grayscale_to_rgb_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.grayscale_to_rgb' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.grayscale_to_rgb'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.grayscale_to_rgb', generated_inputs['tf.image.grayscale_to_rgb'], lib="tf", suffix=0)
