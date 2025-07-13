
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_adjust_saturation_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D image, positive scale
    images = np.array([[[0.0, 0.5, 1.0], [0.2, 0.7, 0.3]], [[0.4, 0.9, 0.6], [0.6, 0.1, 0.8]]], dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    name = None
    input_dict = {"images": images, "scale": scale, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 3D image, negative scale
    images = np.array([[[0.0, 0.5, 1.0], [0.2, 0.7, 0.3]], [[0.4, 0.9, 0.6], [0.6, 0.1, 0.8]]], dtype=np.float32)
    scale = np.array(-0.5, dtype=np.float32)
    name = "negative_scale"
    input_dict = {"images": images, "scale": scale, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D image, scale of 1
    images = np.random.rand(2, 2, 2, 3).astype(np.float32)
    scale = np.array(1.0, dtype=np.float32)
    name = "scale_one"
    input_dict = {"images": images, "scale": scale, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D half image, scale > 1
    images = np.random.rand(2, 2, 3).astype(np.float16)
    scale = np.array(2.0, dtype=np.float32)
    name = "scale_greater_than_one"
    input_dict = {"images": images, "scale": scale, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D image with 0-1 values, scale 0
    images = np.array([[[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]], [[0.1, 0.3, 0.5], [0.7, 0.9, 0.0]]], dtype=np.float32)
    scale = np.array(0.0, dtype=np.float32)
    name = "scale_zero"
    input_dict = {"images": images, "scale": scale, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different image shape
    images = np.random.rand(3, 4, 3).astype(np.float32)
    scale = np.array(0.75, dtype=np.float32)
    name = "diff_shape"
    input_dict = {"images": images, "scale": scale, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Half precision image and scale
    images = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.0, 0.1, 0.2]]], dtype=np.float16)
    scale = np.array(0.3, dtype=np.float32)
    name = "half_precision"
    input_dict = {"images": images, "scale": scale, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D image
    images = np.random.rand(1, 2, 2, 2, 3).astype(np.float32)
    scale = np.array(0.6, dtype=np.float32)
    name = "5d_image"
    input_dict = {"images": images, "scale": scale, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Images with all zeros
    images = np.zeros((2, 2, 3), dtype=np.float32)
    scale = np.array(0.8, dtype=np.float32)
    name = "zero_images"
    input_dict = {"images": images, "scale": scale, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Images with all ones
    images = np.ones((2, 2, 3), dtype=np.float32)
    scale = np.array(0.9, dtype=np.float32)
    name = "ones_images"
    input_dict = {"images": images, "scale": scale, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AdjustSaturation"] = tf_raw_ops_adjust_saturation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AdjustSaturation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AdjustSaturation'.")

check_valid('tf.raw_ops.AdjustSaturation', generated_inputs['tf.raw_ops.AdjustSaturation'], lib="tf", suffix=0)
