
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_resize_bilinear_inputs():
    list_of_inputs = []

    # Input 1
    images = np.random.randint(0, 256, size=(1, 32, 32, 3), dtype=np.int32)
    size = np.array([64, 64], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = "resize_bilinear_1"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "half_pixel_centers": half_pixel_centers, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    images = np.random.rand(2, 16, 16, 1).astype(np.float32)
    size = np.array([32, 32], dtype=np.int32)
    align_corners = True
    half_pixel_centers = False
    name = "resize_bilinear_2"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "half_pixel_centers": half_pixel_centers, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    images = np.random.randint(0, 100, size=(4, 8, 8, 3), dtype=np.int8)
    size = np.array([16, 16], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True
    name = "resize_bilinear_3"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "half_pixel_centers": half_pixel_centers, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    images = np.random.rand(1, 64, 64, 1).astype(np.float64)
    size = np.array([128, 128], dtype=np.int32)
    align_corners = True
    half_pixel_centers = True
    name = "resize_bilinear_4"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "half_pixel_centers": half_pixel_centers, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    images = np.random.randint(0, 256, size=(1, 10, 10, 3), dtype=np.uint8)
    size = np.array([20, 20], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = "resize_bilinear_5"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "half_pixel_centers": half_pixel_centers, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    images = np.random.rand(2, 20, 30, 1).astype(np.float32)
    size = np.array([40, 60], dtype=np.int32)
    align_corners = True
    half_pixel_centers = False
    name = "resize_bilinear_6"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "half_pixel_centers": half_pixel_centers, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    images = np.random.randint(0, 100, size=(4, 5, 5, 3), dtype=np.int16)
    size = np.array([10, 10], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True
    name = "resize_bilinear_7"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "half_pixel_centers": half_pixel_centers, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    images = np.random.rand(1, 25, 25, 1).astype(np.float64)
    size = np.array([50, 50], dtype=np.int32)
    align_corners = True
    half_pixel_centers = True
    name = "resize_bilinear_8"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "half_pixel_centers": half_pixel_centers, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    images = np.random.randint(0, 256, size=(1, 4, 4, 3), dtype=np.uint16)
    size = np.array([8, 8], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = "resize_bilinear_9"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "half_pixel_centers": half_pixel_centers, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    images = np.random.rand(2, 3, 3, 1).astype(np.float32)
    size = np.array([6, 6], dtype=np.int32)
    align_corners = True
    half_pixel_centers = False
    name = "resize_bilinear_10"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "half_pixel_centers": half_pixel_centers, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    images = np.random.rand(1, 20, 20, 1).astype(np.bfloat16)
    size = np.array([40, 40], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True
    name = "resize_bilinear_11"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "half_pixel_centers": half_pixel_centers, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    images = np.random.rand(1, 10, 10, 1).astype(np.half)
    size = np.array([20, 20], dtype=np.int32)
    align_corners = True
    half_pixel_centers = True
    name = "resize_bilinear_12"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "half_pixel_centers": half_pixel_centers, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13
    images = np.random.randint(0, 100, size=(4, 2, 2, 3), dtype=np.int64)
    size = np.array([4, 4], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True
    name = "resize_bilinear_13"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "half_pixel_centers": half_pixel_centers, "name": name}
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
