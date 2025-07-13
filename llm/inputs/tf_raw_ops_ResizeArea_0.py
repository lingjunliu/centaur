
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ResizeArea_inputs():
    list_of_inputs = []

    # Input 1
    images = np.random.rand(1, 10, 10, 3).astype(np.float32)
    size = np.array([5, 5], dtype=np.int32)
    align_corners = False
    name = "resize_area_1"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    images = np.random.rand(2, 20, 30, 1).astype(np.uint8)
    size = np.array([10, 15], dtype=np.int32)
    align_corners = True
    name = "resize_area_2"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    images = np.random.rand(1, 5, 5, 3).astype(np.int32)
    size = np.array([10, 10], dtype=np.int32)
    align_corners = False
    name = "resize_area_3"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    images = np.random.rand(4, 15, 25, 2).astype(np.float64)
    size = np.array([7, 13], dtype=np.int32)
    align_corners = True
    name = "resize_area_4"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    images = np.random.rand(1, 8, 12, 3).astype(np.int8)
    size = np.array([4, 6], dtype=np.int32)
    align_corners = False
    name = "resize_area_5"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    images = np.random.rand(2, 16, 16, 4).astype(np.uint16)
    size = np.array([32, 32], dtype=np.int32)
    align_corners = True
    name = "resize_area_6"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    images = np.random.rand(1, 32, 64, 1).astype(np.int64)
    size = np.array([16, 32], dtype=np.int32)
    align_corners = False
    name = "resize_area_7"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    images = np.random.rand(3, 4, 8, 3).astype(np.float16)
    size = np.array([2, 4], dtype=np.int32)
    align_corners = True
    name = "resize_area_8"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    images = np.random.rand(1, 24, 36, 2).astype(np.float32)
    size = np.array([12, 18], dtype=np.int32)
    align_corners = False
    name = "resize_area_9"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    images = np.random.rand(5, 1, 1, 1).astype(np.float32)
    size = np.array([2, 2], dtype=np.int32)
    align_corners = True
    name = "resize_area_10"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ResizeArea' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ResizeArea'.")

check_valid('tf.raw_ops.ResizeArea', generated_inputs['tf.raw_ops.ResizeArea'], lib="tf", suffix=0)
