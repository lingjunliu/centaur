
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_crop_inputs():
    list_of_inputs = []

    # Input 1
    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    size = np.array([1, 3], dtype=np.int32)
    seed = 123
    name = "crop_1"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    size = np.array([1, 2, 3], dtype=np.int32)
    seed = 456
    name = "crop_2"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    size = np.array([2, 2, 1], dtype=np.int32)
    seed = 789
    name = "crop_3"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.random.rand(10, 10).astype(np.float32)
    size = np.array([5, 5], dtype=np.int32)
    seed = 101
    name = "crop_4"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.random.rand(20, 30, 3).astype(np.float32)
    size = np.array([15, 20, 3], dtype=np.int32)
    seed = 202
    name = "crop_5"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    value = np.arange(1, 28).reshape(3, 3, 3)
    size = np.array([2, 2, 3], dtype=np.int32)
    seed = 303
    name = "crop_6"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    value = np.arange(1, 65).reshape(4, 4, 4)
    size = np.array([2, 2, 2], dtype=np.int32)
    seed = 404
    name = "crop_7"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    value = np.arange(1, 126).reshape(5, 5, 5)
    size = np.array([3, 3, 3], dtype=np.int32)
    seed = 505
    name = "crop_8"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    value = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int32)
    size = np.array([2, 3], dtype=np.int32)
    seed = 606
    name = "crop_9"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.array([[[1, 2], [3, 4], [5,6]], [[7, 8], [9, 10], [11,12]]], dtype=np.int32)
    size = np.array([1, 2, 2], dtype=np.int32)
    seed = 707
    name = "crop_10"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.random_crop"] = tf_image_random_crop_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.random_crop' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_crop'.")

check_valid('tf.image.random_crop', generated_inputs['tf.image.random_crop'], lib="tf", suffix=0)
