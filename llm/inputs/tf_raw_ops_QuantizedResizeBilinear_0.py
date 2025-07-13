
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantized_resize_bilinear_inputs():
    list_of_inputs = []

    # Input 1, valid
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint8)
    size = np.array([1, 1], dtype=np.int32)
    min = np.array(0.0, dtype=np.float32)
    max = np.array(1.0, dtype=np.float32)
    align_corners = False
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "min": min,
        "max": max,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": "resize_bilinear_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid, align_corners = True
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint8)
    size = np.array([3, 3], dtype=np.int32)
    min = np.array(0.0, dtype=np.float32)
    max = np.array(1.0, dtype=np.float32)
    align_corners = True
    half_pixel_centers = False
    input_dict = {
        "images": images,
        "size": size,
        "min": min,
        "max": max,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": "resize_bilinear_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, half_pixel_centers = True
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint8)
    size = np.array([3, 3], dtype=np.int32)
    min = np.array(0.0, dtype=np.float32)
    max = np.array(1.0, dtype=np.float32)
    align_corners = False
    half_pixel_centers = True

    input_dict = {
        "images": images,
        "size": size,
        "min": min,
        "max": max,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": "resize_bilinear_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, different size, different range
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint8)
    size = np.array([5, 5], dtype=np.int32)
    min = np.array(-1.0, dtype=np.float32)
    max = np.array(2.0, dtype=np.float32)
    align_corners = False
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "min": min,
        "max": max,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": "resize_bilinear_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, qint32 type
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int32)
    size = np.array([2, 2], dtype=np.int32)
    min = np.array(-1.0, dtype=np.float32)
    max = np.array(2.0, dtype=np.float32)
    align_corners = False
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "min": min,
        "max": max,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": "resize_bilinear_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, float32 type
    images = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    size = np.array([4, 4], dtype=np.int32)
    min = np.array(-1.0, dtype=np.float32)
    max = np.array(2.0, dtype=np.float32)
    align_corners = False
    half_pixel_centers = False
    input_dict = {
        "images": images,
        "size": size,
        "min": min,
        "max": max,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": "resize_bilinear_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, larger image size
    images = np.random.rand(1, 10, 10, 3).astype(np.float32)
    size = np.array([20, 20], dtype=np.int32)
    min = np.array(0.0, dtype=np.float32)
    max = np.array(1.0, dtype=np.float32)
    align_corners = False
    half_pixel_centers = False
    input_dict = {
        "images": images,
        "size": size,
        "min": min,
        "max": max,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": "resize_bilinear_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, batch size > 1
    images = np.random.rand(2, 5, 5, 1).astype(np.float32)
    size = np.array([10, 10], dtype=np.int32)
    min = np.array(-0.5, dtype=np.float32)
    max = np.array(1.5, dtype=np.float32)
    align_corners = True
    half_pixel_centers = True
    input_dict = {
        "images": images,
        "size": size,
        "min": min,
        "max": max,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": "resize_bilinear_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9, channels > 1
    images = np.random.rand(1, 5, 5, 3).astype(np.float32)
    size = np.array([8, 8], dtype=np.int32)
    min = np.array(0.0, dtype=np.float32)
    max = np.array(1.0, dtype=np.float32)
    align_corners = False
    half_pixel_centers = True
    input_dict = {
        "images": images,
        "size": size,
        "min": min,
        "max": max,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": "resize_bilinear_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, quint8 type, align_corners=True, half_pixel_centers=True
    images = np.array([[[[10], [20]], [[30], [40]]]], dtype=np.uint8)
    size = np.array([4, 4], dtype=np.int32)
    min = np.array(0.0, dtype=np.float32)
    max = np.array(50.0, dtype=np.float32)
    align_corners = True
    half_pixel_centers = True
    input_dict = {
        "images": images,
        "size": size,
        "min": min,
        "max": max,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": "resize_bilinear_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_quantized_resize_bilinear_inputs()
generated_inputs["tf.raw_ops.QuantizedResizeBilinear"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.QuantizedResizeBilinear"].append(
        {
        "images": input_dict["images"],
        "size": input_dict["size"],
        "min": input_dict["min"],
        "max": input_dict["max"],
        "align_corners": input_dict["align_corners"],
        "half_pixel_centers": input_dict["half_pixel_centers"],
        "name": input_dict["name"]
    })

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedResizeBilinear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedResizeBilinear'.")

check_valid('tf.raw_ops.QuantizedResizeBilinear', generated_inputs['tf.raw_ops.QuantizedResizeBilinear'], lib="tf", suffix=0)
