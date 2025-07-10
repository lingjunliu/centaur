
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_flip_up_down_inputs():
    list_of_inputs = []

    # Input 1: 3D image, seed
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)
    seed = np.array([2, 3], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4D image, seed
    image = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
    seed = np.array([1, 4], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D image, seed with int64
    image = np.array([[[10], [20]], [[30], [40]]], dtype=np.uint8)
    seed = np.array([5, 6], dtype=np.int64)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D image with more channels, seed
    image = np.random.rand(1, 32, 32, 3).astype(np.float64)
    seed = np.array([7, 8], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D image with different height and width, seed
    image = np.random.rand(64, 128, 1).astype(np.float32)
    seed = np.array([9, 10], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: 4D image with different height and width, seed
    image = np.random.rand(2, 64, 128, 3).astype(np.float32)
    seed = np.array([11, 12], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D image with complex numbers, seed
    image = np.array([[[1+1j], [2+2j]], [[3+3j], [4+4j]]], dtype=np.complex64)
    seed = np.array([13, 14], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D image, seed with negative values
    image = np.random.rand(1, 32, 32, 3).astype(np.float32)
    seed = np.array([-1, -2], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D image with large values, seed
    image = np.array([[[1000], [2000]], [[3000], [4000]]], dtype=np.int32)
    seed = np.array([15, 16], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D image, seed with large int64 values
    image = np.random.rand(1, 32, 32, 3).astype(np.float32)
    seed = np.array([2**31-1, 2**31-2], dtype=np.int64)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D image with zero values, seed
    image = np.array([[[0], [0]], [[0], [0]]], dtype=np.int32)
    seed = np.array([17, 18], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.stateless_random_flip_up_down"] = tf_image_stateless_random_flip_up_down_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.stateless_random_flip_up_down' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_flip_up_down'.")

check_valid('tf.image.stateless_random_flip_up_down', generated_inputs['tf.image.stateless_random_flip_up_down'], lib="tf", suffix=0)
