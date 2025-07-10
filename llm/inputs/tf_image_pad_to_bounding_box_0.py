
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_pad_to_bounding_box_inputs():
    list_of_inputs = []

    # Input 1: 3D image, simple padding
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    offset_height = 1
    offset_width = 1
    target_height = 4
    target_width = 4
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4D image, batch size 1
    image = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.float32)
    offset_height = 0
    offset_width = 0
    target_height = 3
    target_width = 3
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D image, no padding needed
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    offset_height = 0
    offset_width = 0
    target_height = 2
    target_width = 3
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D image, larger padding
    image = np.array([[[[1, 2, 3]]]], dtype=np.float32)
    offset_height = 2
    offset_width = 2
    target_height = 5
    target_width = 5
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D image, unequal padding
    image = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
    offset_height = 1
    offset_width = 2
    target_height = 5
    target_width = 6
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D image, multiple batches
    image = np.array([[[[1, 2, 3], [4, 5, 6]]], [[[7, 8, 9], [10, 11, 12]]]], dtype=np.float32)
    offset_height = 1
    offset_width = 1
    target_height = 4
    target_width = 4
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D, different number of channels
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.float32)
    offset_height = 1
    offset_width = 1
    target_height = 4
    target_width = 4
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D, multiple channels, batches
    image = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.float32)
    offset_height = 0
    offset_width = 0
    target_height = 3
    target_width = 3
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Remove invalid input
    #image = np.array([[[1], [2]], [[3], [4]]], dtype=np.float32)
    #offset_height = 1
    #offset_width = 1
    #target_height = 4
    #target_width = 4
    #input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D, different batch and channel sizes
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float32)
    offset_height = 1
    offset_width = 1
    target_height = 5
    target_width = 5
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.pad_to_bounding_box"] = tf_image_pad_to_bounding_box_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.pad_to_bounding_box' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.pad_to_bounding_box'.")

check_valid('tf.image.pad_to_bounding_box', generated_inputs['tf.image.pad_to_bounding_box'], lib="tf", suffix=0)
