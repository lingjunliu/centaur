
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_crop_to_bounding_box_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D image
    image = np.arange(1, 28, dtype=np.float32).reshape((3, 3, 3))
    offset_height = 0
    offset_width = 0
    target_height = 2
    target_width = 2
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4D image, batch size 2
    image = np.arange(1, 55, dtype=np.float32).reshape((2, 3, 3, 3))
    offset_height = 1
    offset_width = 1
    target_height = 1
    target_width = 1
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different offsets and target sizes
    image = np.arange(1, 101, dtype=np.float32).reshape((10, 10, 1))
    offset_height = 2
    offset_width = 3
    target_height = 5
    target_width = 4
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Single channel image
    image = np.arange(1, 17, dtype=np.float32).reshape((4, 4, 1))
    offset_height = 0
    offset_width = 0
    target_height = 4
    target_width = 4
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger image, small crop
    image = np.arange(1, 226, dtype=np.float32).reshape((15, 15, 1))
    offset_height = 7
    offset_width = 7
    target_height = 3
    target_width = 3
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D, different batch size and channel
    image = np.arange(1, 61, dtype=np.float32).reshape((3, 5, 4, 1))
    offset_height = 1
    offset_width = 1
    target_height = 3
    target_width = 2
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Offset at the edge
    image = np.arange(1, 101, dtype=np.float32).reshape((10, 10, 1))
    offset_height = 9
    offset_width = 9
    target_height = 1
    target_width = 1
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Target is full image
    image = np.arange(1, 101, dtype=np.float32).reshape((10, 10, 1))
    offset_height = 0
    offset_width = 0
    target_height = 10
    target_width = 10
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D, float64
    image = np.arange(1, 28, dtype=np.float64).reshape((3, 3, 3))
    offset_height = 0
    offset_width = 0
    target_height = 2
    target_width = 2
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D, int32
    image = np.arange(1, 55, dtype=np.int32).reshape((2, 3, 3, 3))
    offset_height = 1
    offset_width = 1
    target_height = 1
    target_width = 1
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: larger crop
    image = np.arange(1, 65, dtype=np.float32).reshape((8, 8, 1))
    offset_height = 2
    offset_width = 2
    target_height = 5
    target_width = 5
    input_dict = {"image": image, "offset_height": offset_height, "offset_width": offset_width, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.crop_to_bounding_box"] = tf_image_crop_to_bounding_box_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.crop_to_bounding_box' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.crop_to_bounding_box'.")

check_valid('tf.image.crop_to_bounding_box', generated_inputs['tf.image.crop_to_bounding_box'], lib="tf", suffix=0)
