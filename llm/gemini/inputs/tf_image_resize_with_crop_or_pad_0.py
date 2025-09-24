
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_resize_with_crop_or_pad_inputs():
    list_of_inputs = []

    # Input 1: 3D image, crop
    image = np.arange(75, dtype=np.int32).reshape(5, 5, 3)
    target_height = 3
    target_width = 3
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D image, pad
    image = np.arange(1, 28, dtype=np.int32).reshape(3, 3, 3)
    target_height = 5
    target_width = 5
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D image, crop
    image = np.arange(100, dtype=np.int32).reshape(1, 10, 10, 1)
    target_height = 5
    target_width = 5
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D image, pad
    image = np.arange(36, dtype=np.int32).reshape(1, 6, 6, 1)
    target_height = 10
    target_width = 10
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D image, same size
    image = np.arange(27, dtype=np.int32).reshape(3, 3, 3)
    target_height = 3
    target_width = 3
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D image, same size
    image = np.arange(16, dtype=np.int32).reshape(1, 4, 4, 1)
    target_height = 4
    target_width = 4
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D image, crop height only
    image = np.arange(50, dtype=np.int32).reshape(5, 5, 2)
    target_height = 3
    target_width = 5
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D image, crop width only
    image = np.arange(50, dtype=np.int32).reshape(5, 5, 2)
    target_height = 5
    target_width = 3
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D image, pad height only
    image = np.arange(18, dtype=np.int32).reshape(3, 3, 2)
    target_height = 5
    target_width = 3
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D image, pad width only
    image = np.arange(18, dtype=np.int32).reshape(3, 3, 2)
    target_height = 3
    target_width = 5
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D image, different batch size
    image = np.arange(200, dtype=np.int32).reshape(2, 10, 5, 2)
    target_height = 3
    target_width = 3
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 4D image, crop and pad
    image = np.arange(200, dtype=np.int32).reshape(2, 10, 5, 2)
    target_height = 5
    target_width = 10
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.resize_with_crop_or_pad"] = tf_image_resize_with_crop_or_pad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.resize_with_crop_or_pad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.resize_with_crop_or_pad'.")

check_valid('tf.image.resize_with_crop_or_pad', generated_inputs['tf.image.resize_with_crop_or_pad'], lib="tf", suffix=0)
