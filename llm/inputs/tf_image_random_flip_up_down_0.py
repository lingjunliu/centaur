
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_flip_up_down_inputs():
    list_of_inputs = []

    # Input 1: 3D image with seed
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    seed = 10
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4D batch of images with seed
    image = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                      [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]], dtype=np.float32)
    seed = 20
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D image with seed=None
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    seed = 1234
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D batch of images with different channels
    image = np.random.rand(2, 32, 32, 1).astype(np.float64)
    seed = 30
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D image with small dimensions
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)
    seed = 40
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D batch of images with seed
    image = np.random.randint(0, 256, size=(2, 64, 64, 3), dtype=np.uint8)
    seed = 50
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D image with seed
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13,14,15],[16,17,18]]], dtype=np.int16)
    seed = 60
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D batch of images with seed
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float16)
    seed = 70
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D image with large dimensions
    image = np.random.rand(256, 256, 3).astype(np.float32)
    seed = 80
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D batch of images with seed and large dimensions
    image = np.random.randint(0, 256, size=(4, 128, 128, 1), dtype=np.uint8)
    seed = 90
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.random_flip_up_down"] = tf_image_random_flip_up_down_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.random_flip_up_down' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_flip_up_down'.")

check_valid('tf.image.random_flip_up_down', generated_inputs['tf.image.random_flip_up_down'], lib="tf", suffix=0)
