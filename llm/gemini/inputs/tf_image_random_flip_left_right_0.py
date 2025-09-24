
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_flip_left_right_inputs():
    list_of_inputs = []

    # Input 1: 3D image, positive seed
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)
    seed = 5
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4D images, zero seed
    images = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.float32)
    seed = 0
    input_dict = {"image": images, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D image, no seed, set tf.random.set_seed
    tf.random.set_seed(1)
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.float64)
    seed = None
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D images, negative seed
    images = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.uint8)
    seed = -1
    input_dict = {"image": images, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D image, large seed
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.int64)
    seed = 2**31 - 1
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D images, seed is a string representation of an integer
    images = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.int8)
    seed = 10
    input_dict = {"image": images, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D image, shape (1,2,1)
    image = np.array([[[1], [2]]], dtype=np.int32)
    seed = 1
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D images, batch size 1, shape (1,1,2,1)
    images = np.array([[[[1], [2]]]], dtype=np.float32)
    seed = 2
    input_dict = {"image": images, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D image with multiple channels
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    seed = 3
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D images with multiple channels and batch
    images = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float32)
    seed = 4
    input_dict = {"image": images, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs["tf.image.random_flip_left_right"] = tf_image_random_flip_left_right_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.random_flip_left_right' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_flip_left_right'.")

check_valid('tf.image.random_flip_left_right', generated_inputs['tf.image.random_flip_left_right'], lib="tf", suffix=0)
