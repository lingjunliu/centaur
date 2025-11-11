
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_random_flip_left_right_inputs():
    list_of_inputs = []

    image1 = np.array([[[1], [2]], [[3], [4]]])
    seed1 = 5
    input_dict1 = {"image": image1, "seed": seed1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    seed2 = 6
    input_dict2 = {"image": image2, "seed": seed2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]])
    seed3 = 7
    input_dict3 = {"image": image3, "seed": seed3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
    seed4 = 8
    input_dict4 = {"image": image4, "seed": seed4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.array([[[1, 2, 3], [4, 5, 6]]])
    seed5 = 9
    input_dict5 = {"image": image5, "seed": seed5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]])
    seed6 = 10
    input_dict6 = {"image": image6, "seed": seed6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = np.array([[[[1], [2]], [[3], [4]]]])
    seed7 = -1
    input_dict7 = {"image": image7, "seed": seed7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    image8 = np.array([[[1, 2], [3, 4], [5, 6]]])
    seed8 = 0
    input_dict8 = {"image": image8, "seed": seed8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    image9 = np.array([[[[1, 2, 3]], [[4, 5, 6]]]])
    seed9 = 11
    input_dict9 = {"image": image9, "seed": seed9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    image10 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    seed10 = 12
    input_dict10 = {"image": image10, "seed": seed10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.random_flip_left_right"] = tf_image_random_flip_left_right_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.random_flip_left_right' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_flip_left_right'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.random_flip_left_right', generated_inputs['tf.image.random_flip_left_right'], lib="tf", suffix=0)
