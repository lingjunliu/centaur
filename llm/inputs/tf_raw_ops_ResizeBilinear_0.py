
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_resize_bilinear_inputs():
    list_of_inputs = []

    # Input 1: Basic case with uint8 images
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint8)
    size = np.array([4, 4], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = None

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different size, align_corners=True
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint8)
    size = np.array([3, 3], dtype=np.int32)
    align_corners = True
    half_pixel_centers = False
    name = None

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: half_pixel_centers=True
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint8)
    size = np.array([3, 3], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True
    name = None

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int32 images
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int32)
    size = np.array([4, 4], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = None

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32 images
    images = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    size = np.array([4, 4], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = None

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiple channels
    images = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.uint8)
    size = np.array([4, 4], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = None

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batch size > 1
    images = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.uint8)
    size = np.array([4, 4], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = None

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Smaller size
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint8)
    size = np.array([1, 1], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = None

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int16 images
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int16)
    size = np.array([4, 4], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = None

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64 images
    images = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float64)
    size = np.array([4, 4], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = None

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ResizeBilinear"] = tf_raw_ops_resize_bilinear_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ResizeBilinear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ResizeBilinear'.")

check_valid('tf.raw_ops.ResizeBilinear', generated_inputs['tf.raw_ops.ResizeBilinear'], lib="tf", suffix=0)
