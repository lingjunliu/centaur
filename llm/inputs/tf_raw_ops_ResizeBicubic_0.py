
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_resize_bicubic_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    images = np.random.rand(1, 32, 32, 3).astype(np.float32)
    size = np.array([64, 64], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: align_corners = True
    images = np.random.rand(1, 32, 32, 3).astype(np.float32)
    size = np.array([64, 64], dtype=np.int32)
    align_corners = True
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: half_pixel_centers = True
    images = np.random.rand(1, 32, 32, 3).astype(np.float32)
    size = np.array([64, 64], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different image size
    images = np.random.rand(1, 64, 64, 3).astype(np.float32)
    size = np.array([32, 32], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different number of channels
    images = np.random.rand(1, 32, 32, 1).astype(np.float32)
    size = np.array([64, 64], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batch size > 1
    images = np.random.rand(4, 32, 32, 3).astype(np.float32)
    size = np.array([64, 64], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different image type (int8)
    images = np.random.randint(-128, 127, size=(1, 32, 32, 3), dtype=np.int8)
    size = np.array([64, 64], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different image type (uint8)
    images = np.random.randint(0, 255, size=(1, 32, 32, 3), dtype=np.uint8)
    size = np.array([64, 64], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Different size
    images = np.random.rand(1, 32, 32, 3).astype(np.float32)
    size = np.array([128, 256], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half_pixel_centers = True, align_corners = False
    images = np.random.rand(1, 32, 32, 3).astype(np.float32)
    size = np.array([64, 64], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True

    input_dict = {
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ResizeBicubic"] = tf_raw_ops_resize_bicubic_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ResizeBicubic' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ResizeBicubic'.")

check_valid('tf.raw_ops.ResizeBicubic', generated_inputs['tf.raw_ops.ResizeBicubic'], lib="tf", suffix=0)
