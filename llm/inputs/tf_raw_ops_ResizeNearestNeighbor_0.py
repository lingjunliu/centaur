
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_resize_nearest_neighbor_inputs():
    list_of_inputs = []

    # Input 1
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
    size = np.array([1, 1], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = "resize_1"

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    images = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    size = np.array([3, 3], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = "resize_2"

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    images = np.array([[[[1, 2, 3]]]], dtype=np.float32)
    size = np.array([5, 5], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True
    name = "resize_3"

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int32)
    size = np.array([4, 4], dtype=np.int32)
    align_corners = True
    half_pixel_centers = False
    name = "resize_4"

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    images = np.array([[[[1.5], [2.5]], [[3.5], [4.5]]]], dtype=np.float64)
    size = np.array([2, 3], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = "resize_5"

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    images = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.float32)
    size = np.array([5, 2], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = "resize_6"

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Removed problematic input

    # Input 8
    images = np.array([[[[1, 2, 3, 4]]]], dtype=np.float32)
    size = np.array([10, 10], dtype=np.int32)
    align_corners = True
    half_pixel_centers = False
    name = "resize_8"

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint8)
    size = np.array([7, 3], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True
    name = "resize_9"

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int8)
    size = np.array([8, 4], dtype=np.int32)
    align_corners = True
    half_pixel_centers = False
    name = "resize_10"

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
    size = np.array([2, 2], dtype=np.int32)
    align_corners = True
    half_pixel_centers = False
    name = "resize_11"

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
generated_inputs["tf.raw_ops.ResizeNearestNeighbor"] = tf_raw_ops_resize_nearest_neighbor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ResizeNearestNeighbor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ResizeNearestNeighbor'.")

check_valid('tf.raw_ops.ResizeNearestNeighbor', generated_inputs['tf.raw_ops.ResizeNearestNeighbor'], lib="tf", suffix=0)
