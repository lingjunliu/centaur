
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_resize_area_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    images = np.random.rand(1, 10, 10, 3).astype(np.float32)
    size = np.array([5, 5], dtype=np.int32)
    align_corners = False
    name = None
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different image size and data type
    images = np.random.randint(0, 256, size=(2, 20, 30, 1), dtype=np.uint8)
    size = np.array([10, 15], dtype=np.int32)
    align_corners = True
    name = "resize_area_op"
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch size > 1
    images = np.random.rand(4, 8, 12, 3).astype(np.float64)
    size = np.array([4, 6], dtype=np.int32)
    align_corners = False
    name = None
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small image size
    images = np.random.rand(1, 2, 2, 3).astype(np.float32)
    size = np.array([1, 1], dtype=np.int32)
    align_corners = False
    name = None
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Int data type
    images = np.random.randint(0, 100, size=(1, 16, 16, 1), dtype=np.int32)
    size = np.array([8, 8], dtype=np.int32)
    align_corners = True
    name = None
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Half data type
    images = np.random.rand(1, 10, 10, 3).astype(np.float16)
    size = np.array([5, 5], dtype=np.int32)
    align_corners = False
    name = None
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large resize
    images = np.random.rand(1, 5, 5, 3).astype(np.float32)
    size = np.array([20, 20], dtype=np.int32)
    align_corners = True
    name = None
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int8 data type
    images = np.random.randint(0, 127, size=(1, 16, 16, 1), dtype=np.int8)
    size = np.array([8, 8], dtype=np.int32)
    align_corners = True
    name = None
    input_dict = {"images": images, "size": size, "align_corners": align_corners, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ResizeArea"] = tf_raw_ops_resize_area_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ResizeArea' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ResizeArea'.")

check_valid('tf.raw_ops.ResizeArea', generated_inputs['tf.raw_ops.ResizeArea'], lib="tf", suffix=0)
