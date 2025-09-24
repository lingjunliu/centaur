
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_resize_with_pad_inputs():
    list_of_inputs = []

    # Input 1
    image = np.random.rand(100, 100, 3).astype(np.float32)
    target_height = 200
    target_width = 300
    method = 'bilinear'
    antialias = False
    input_dict = {'image': image, 'target_height': target_height, 'target_width': target_width, 'method': method, 'antialias': antialias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image = np.random.rand(50, 50, 1).astype(np.float32)
    target_height = 100
    target_width = 100
    method = 'nearest'
    antialias = False
    input_dict = {'image': image, 'target_height': target_height, 'target_width': target_width, 'method': method, 'antialias': antialias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image = np.random.rand(1, 64, 64, 3).astype(np.float32)
    target_height = 128
    target_width = 256
    method = 'bicubic'
    antialias = True
    input_dict = {'image': image, 'target_height': target_height, 'target_width': target_width, 'method': method, 'antialias': antialias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image = np.random.rand(4, 32, 32, 3).astype(np.float32)
    target_height = 64
    target_width = 64
    method = 'area'
    antialias = False
    input_dict = {'image': image, 'target_height': target_height, 'target_width': target_width, 'method': method, 'antialias': antialias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    image = np.random.rand(1, 128, 128, 3).astype(np.float32)
    target_height = 64
    target_width = 32
    method = 'lanczos3'
    antialias = True
    input_dict = {'image': image, 'target_height': target_height, 'target_width': target_width, 'method': method, 'antialias': antialias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    image = np.random.rand(32, 32, 3).astype(np.float32)
    target_height = 32
    target_width = 32
    method = 'bilinear'
    antialias = False
    input_dict = {'image': image, 'target_height': target_height, 'target_width': target_width, 'method': method, 'antialias': antialias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    image = np.random.rand(1, 256, 256, 1).astype(np.float32)
    target_height = 512
    target_width = 128
    method = 'nearest'
    antialias = True
    input_dict = {'image': image, 'target_height': target_height, 'target_width': target_width, 'method': method, 'antialias': antialias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    image = np.random.rand(2, 64, 64, 4).astype(np.float32)
    target_height = 32
    target_width = 128
    method = 'bicubic'
    antialias = False
    input_dict = {'image': image, 'target_height': target_height, 'target_width': target_width, 'method': method, 'antialias': antialias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    image = np.random.rand(128, 64, 3).astype(np.float32)
    target_height = 256
    target_width = 128
    method = 'area'
    antialias = True
    input_dict = {'image': image, 'target_height': target_height, 'target_width': target_width, 'method': method, 'antialias': antialias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    image = np.random.rand(1, 16, 16, 3).astype(np.float32)
    target_height = 32
    target_width = 64
    method = 'lanczos3'
    antialias = False
    input_dict = {'image': image, 'target_height': target_height, 'target_width': target_width, 'method': method, 'antialias': antialias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.resize_with_pad"] = tf_image_resize_with_pad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.resize_with_pad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.resize_with_pad'.")

check_valid('tf.image.resize_with_pad', generated_inputs['tf.image.resize_with_pad'], lib="tf", suffix=0)
