
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_crop_inputs():
    list_of_inputs = []

    # Input 1
    value = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    size = np.array([1, 2, 3], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    name = "crop1"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    size = np.array([2, 1, 3], dtype=np.int32)
    seed = np.array([3, 4], dtype=np.int32)
    name = "crop2"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.int32)
    size = np.array([50, 50, 3], dtype=np.int32)
    seed = np.array([5, 6], dtype=np.int32)
    name = "crop3"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.random.rand(28, 28, 1).astype(np.float32)
    size = np.array([20, 20, 1], dtype=np.int32)
    seed = np.array([7, 8], dtype=np.int32)
    name = "crop4"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.random.rand(32, 32, 3).astype(np.float32)
    size = np.array([32, 32, 3], dtype=np.int32)
    seed = np.array([9, 10], dtype=np.int32)
    name = "crop5"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    value = np.arange(24).reshape((2, 3, 4)).astype(np.int32)
    size = np.array([1, 2, 3], dtype=np.int32)
    seed = np.array([11, 12], dtype=np.int32)
    name = "crop6"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    value = np.arange(16).reshape((4, 4)).astype(np.int32)
    size = np.array([2, 2], dtype=np.int32)
    seed = np.array([13, 14], dtype=np.int32)
    name = "crop7"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    value = np.random.randint(0, 10, size=(5, 5, 5), dtype=np.int32)
    size = np.array([3, 3, 3], dtype=np.int32)
    seed = np.array([15, 16], dtype=np.int32)
    name = "crop8"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    value = np.random.rand(10, 10).astype(np.float32)
    size = np.array([5, 5], dtype=np.int32)
    seed = np.array([17, 18], dtype=np.int32)
    name = "crop9"
    input_dict = {"value": value, "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    size = np.array([32, 32, 3], dtype=np.int32)
    seed = np.array([19, 20], dtype=np.int32)
    name = "crop10"
    input_dict = {"value": value.astype(np.int32), "size": size, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.stateless_random_crop"] = tf_image_stateless_random_crop_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.stateless_random_crop' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_crop'.")

check_valid('tf.image.stateless_random_crop', generated_inputs['tf.image.stateless_random_crop'], lib="tf", suffix=0)
