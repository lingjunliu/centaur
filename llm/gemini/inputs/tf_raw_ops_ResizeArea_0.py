
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ResizeArea_inputs():
    list_of_inputs = []

    # Input 1: Float32 input, downsampling
    images = np.random.rand(1, 10, 10, 3).astype(np.float32)
    size = np.array([5, 5], dtype=np.int32)
    align_corners = False
    name = "resize_1"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 2: UInt8 input, downsampling with align_corners=True
    images = np.random.randint(0, 256, size=(2, 8, 8, 1)).astype(np.uint8)
    size = np.array([4, 4], dtype=np.int32)
    align_corners = True
    name = "resize_2"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 3: Int32 input with negative values
    images = np.random.randint(-100, 100, size=(1, 16, 16, 4)).astype(np.int32)
    size = np.array([8, 8], dtype=np.int32)
    align_corners = False
    name = "resize_3"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 4: Float64 input
    images = np.random.rand(3, 12, 12, 3).astype(np.float64)
    size = np.array([6, 6], dtype=np.int32)
    align_corners = True
    name = "resize_4"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 5: Float32 input, other dimensions
    images = np.random.rand(1, 20, 20, 2).astype(np.float32)
    size = np.array([10, 10], dtype=np.int32)
    align_corners = False
    name = "resize_5"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 6: UInt8 input with multiple channels
    images = np.random.randint(0, 256, size=(2, 6, 6, 3)).astype(np.uint8)
    size = np.array([3, 3], dtype=np.int32)
    align_corners = True
    name = "resize_6"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 7: Int32 input
    images = np.random.randint(-50, 50, size=(1, 4, 4, 1)).astype(np.int32)
    size = np.array([2, 2], dtype=np.int32)
    align_corners = False
    name = "resize_7"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 8: Float64 input with larger batch
    images = np.random.rand(4, 14, 14, 3).astype(np.float64)
    size = np.array([7, 7], dtype=np.int32)
    align_corners = True
    name = "resize_8"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 9: Float32 input, scaling to same size
    images = np.random.rand(1, 22, 22, 3).astype(np.float32)
    size = np.array([22, 22], dtype=np.int32)
    align_corners = False
    name = "resize_9"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 10: Float32 input, upsampling
    images = np.random.rand(2, 2, 2, 2).astype(np.float32)
    size = np.array([4, 4], dtype=np.int32)
    align_corners = True
    name = "resize_10"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.ResizeArea"] = tf_raw_ops_ResizeArea_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ResizeArea' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ResizeArea'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ResizeArea', generated_inputs['tf.raw_ops.ResizeArea'], lib="tf", suffix=0)
