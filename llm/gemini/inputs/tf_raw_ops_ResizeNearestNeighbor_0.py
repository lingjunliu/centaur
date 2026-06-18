
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ResizeNearestNeighbor_inputs():
    list_of_inputs = []

    # Input 1
    images = np.random.rand(1, 2, 2, 3).astype(np.float32)
    size = np.array([4, 4], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = "resize_1"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 2
    images = np.random.randint(0, 255, size=(2, 4, 4, 1)).astype(np.int32)
    size = np.array([2, 2], dtype=np.int32)
    align_corners = True
    half_pixel_centers = False
    name = "resize_2"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 3
    images = np.random.rand(1, 10, 10, 3).astype(np.float64)
    size = np.array([5, 5], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True
    name = "resize_3"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 4
    images = np.random.randint(-128, 127, size=(4, 8, 8, 4)).astype(np.int32)
    size = np.array([16, 16], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True
    name = "resize_4"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 5
    images = np.random.rand(1, 3, 3, 2).astype(np.float32)
    size = np.array([6, 6], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = "resize_5"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 6
    images = np.random.randint(-1000, 1000, size=(2, 5, 5, 3)).astype(np.int32)
    size = np.array([10, 10], dtype=np.int32)
    align_corners = True
    half_pixel_centers = False
    name = "resize_6"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 7
    images = np.random.randint(-1000, 1000, size=(1, 2, 2, 1)).astype(np.int32)
    size = np.array([1, 1], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True
    name = "resize_7"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 8
    images = np.random.rand(1, 4, 4, 3).astype(np.float64)
    size = np.array([8, 8], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True
    name = "resize_8"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 9
    images = np.random.rand(1, 3, 3, 3).astype(np.float32)
    size = np.array([10, 10], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = "resize_9"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 10
    images = np.random.rand(2, 6, 6, 1).astype(np.float32)
    size = np.array([3, 3], dtype=np.int32)
    align_corners = True
    half_pixel_centers = False
    name = "resize_10"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.ResizeNearestNeighbor"] = tf_raw_ops_ResizeNearestNeighbor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ResizeNearestNeighbor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ResizeNearestNeighbor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ResizeNearestNeighbor', generated_inputs['tf.raw_ops.ResizeNearestNeighbor'], lib="tf", suffix=0)
