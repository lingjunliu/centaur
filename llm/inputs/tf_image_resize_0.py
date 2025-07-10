
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_resize_inputs():
    list_of_inputs = []

    # Input 1
    images = tf.constant(np.random.rand(1, 28, 28, 3)).numpy()
    size = tf.constant([56, 56]).numpy()
    method = 'bilinear'
    preserve_aspect_ratio = False
    antialias = False
    name = None

    input_dict = {
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    images = tf.constant(np.random.rand(32, 64, 64, 1)).numpy()
    size = tf.constant([32, 32]).numpy()
    method = 'nearest'
    preserve_aspect_ratio = True
    antialias = False
    name = 'resize_image'

    input_dict = {
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    images = tf.constant(np.random.rand(1, 128, 128, 3)).numpy()
    size = tf.constant([64, 64]).numpy()
    method = 'lanczos3'
    preserve_aspect_ratio = False
    antialias = True
    name = None

    input_dict = {
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    images = tf.constant(np.random.rand(16, 32, 32, 3)).numpy()
    size = tf.constant([16, 64]).numpy()
    method = 'lanczos5'
    preserve_aspect_ratio = True
    antialias = False
    name = 'image_resize'

    input_dict = {
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    images = tf.constant(np.random.rand(1, 64, 64, 3)).numpy()
    size = tf.constant([128, 32]).numpy()
    method = 'bicubic'
    preserve_aspect_ratio = False
    antialias = True
    name = None

    input_dict = {
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    images = tf.constant(np.random.rand(8, 16, 16, 1)).numpy()
    size = tf.constant([8, 8]).numpy()
    method = 'gaussian'
    preserve_aspect_ratio = True
    antialias = False
    name = 'resize_gauss'

    input_dict = {
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    images = tf.constant(np.random.rand(1, 32, 32, 3)).numpy()
    size = tf.constant([64, 128]).numpy()
    method = 'area'
    preserve_aspect_ratio = False
    antialias = False
    name = None

    input_dict = {
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    images = tf.constant(np.random.rand(4, 64, 64, 1)).numpy()
    size = tf.constant([32, 32]).numpy()
    method = 'mitchellcubic'
    preserve_aspect_ratio = True
    antialias = True
    name = 'resize_mitchell'

    input_dict = {
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D images
    images = tf.constant(np.random.rand(64, 64, 3)).numpy()
    size = tf.constant([128, 128]).numpy()
    method = 'bilinear'
    preserve_aspect_ratio = False
    antialias = False
    name = None
    
    input_dict = {
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: different size, 3D image
    images = tf.constant(np.random.rand(32, 32, 1)).numpy()
    size = tf.constant([64, 16]).numpy()
    method = 'nearest'
    preserve_aspect_ratio = True
    antialias = False
    name = 'resize_nearest'

    input_dict = {
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: tiny image
    images = tf.constant(np.random.rand(1, 2, 2, 3)).numpy()
    size = tf.constant([4, 4]).numpy()
    method = 'bilinear'
    preserve_aspect_ratio = False
    antialias = False
    name = None

    input_dict = {
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.resize"] = tf_image_resize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.resize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.resize'.")

check_valid('tf.image.resize', generated_inputs['tf.image.resize'], lib="tf", suffix=0)
