
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_flip_left_right_inputs():
    list_of_inputs = []

    # Input 1: 3D array
    image1 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    input_dict1 = {"image": image1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 4D array, batch size 1
    image2 = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.float32)
    input_dict2 = {"image": image2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D array with different dimensions
    image3 = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.float32)
    input_dict3 = {"image": image3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D array with batch size > 1
    image4 = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                       [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]], dtype=np.float32)
    input_dict4 = {"image": image4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D array with integer values
    image5 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    input_dict5 = {"image": image5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 4D array with integer values
    image6 = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.int32)
    input_dict6 = {"image": image6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 3D array with a single channel
    image7 = np.array([[[1], [2]], [[3], [4]]], dtype=np.float32)
    input_dict7 = {"image": image7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 4D array with a single channel and batch size > 1
    image8 = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.float32)
    input_dict8 = {"image": image8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: 3D array, smaller size
    image9 = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
    input_dict9 = {"image": image9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: 4D array, smaller size, batch size 1
    image10 = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.float32)
    input_dict10 = {"image": image10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.flip_left_right"] = tf_image_flip_left_right_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.flip_left_right' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.flip_left_right'.")

check_valid('tf.image.flip_left_right', generated_inputs['tf.image.flip_left_right'], lib="tf", suffix=0)
