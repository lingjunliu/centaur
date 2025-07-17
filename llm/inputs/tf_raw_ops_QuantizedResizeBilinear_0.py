
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizedResizeBilinear_inputs():
    list_of_inputs = []

    # Input 1
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint8)
    size = np.array([4, 4], dtype=np.int32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(255.0, dtype=np.float32)
    align_corners = False
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "min": min_val,
        "max": max_val,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    images = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint8)
    size = np.array([2, 2], dtype=np.int32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(10.0, dtype=np.float32)
    align_corners = True
    half_pixel_centers = True
    input_dict = {
        "images": images,
        "size": size,
        "min": min_val,
        "max": max_val,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint8)
    size = np.array([1, 1], dtype=np.int32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(5.0, dtype=np.float32)
    align_corners = False
    half_pixel_centers = True

    input_dict = {
        "images": images,
        "size": size,
        "min": min_val,
        "max": max_val,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    images = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.uint8)
    size = np.array([3, 3], dtype=np.int32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(12.0, dtype=np.float32)
    align_corners = True
    half_pixel_centers = False
    input_dict = {
        "images": images,
        "size": size,
        "min": min_val,
        "max": max_val,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    images = np.array([[[[1]], [[2]]]], dtype=np.uint8)
    size = np.array([5, 5], dtype=np.int32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(2.0, dtype=np.float32)
    align_corners = False
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "min": min_val,
        "max": max_val,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    images = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int32)
    size = np.array([4, 4], dtype=np.int32)
    min_val = np.array(-10.0, dtype=np.float32)
    max_val = np.array(10.0, dtype=np.float32)
    align_corners = True
    half_pixel_centers = True

    input_dict = {
        "images": images,
        "size": size,
        "min": min_val,
        "max": max_val,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    images = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    size = np.array([1, 1], dtype=np.int32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(5.0, dtype=np.float32)
    align_corners = False
    half_pixel_centers = True

    input_dict = {
        "images": images,
        "size": size,
        "min": min_val,
        "max": max_val,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    images = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]], dtype=np.float32)
    size = np.array([3, 3], dtype=np.int32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(12.0, dtype=np.float32)
    align_corners = True
    half_pixel_centers = False
    input_dict = {
        "images": images,
        "size": size,
        "min": min_val,
        "max": max_val,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    images = np.array([[[[1.0]], [[2.0]]]], dtype=np.float32)
    size = np.array([5, 5], dtype=np.int32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(2.0, dtype=np.float32)
    align_corners = False
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "min": min_val,
        "max": max_val,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    images = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int32)
    size = np.array([2, 2], dtype=np.int32)
    min_val = np.array(-5.0, dtype=np.float32)
    max_val = np.array(5.0, dtype=np.float32)
    align_corners = False
    half_pixel_centers = False
    input_dict = {
        "images": images,
        "size": size,
        "min": min_val,
        "max": max_val,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    images = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.uint8)
    size = np.array([1, 1], dtype=np.int32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(255.0, dtype=np.float32)
    align_corners = True
    half_pixel_centers = True

    input_dict = {
        "images": images,
        "size": size,
        "min": min_val,
        "max": max_val,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    # Input 12
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int32)
    size = np.array([2, 2], dtype=np.int32)
    min_val = np.array(-10.0, dtype=np.float32)
    max_val = np.array(10.0, dtype=np.float32)
    align_corners = False
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "min": min_val,
        "max": max_val,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
    size = np.array([2, 2], dtype=np.int32)
    min_val = np.array(-10.0, dtype=np.float32)
    max_val = np.array(10.0, dtype=np.float32)
    align_corners = True
    half_pixel_centers = True
    input_dict = {
        "images": images,
        "size": size,
        "min": min_val,
        "max": max_val,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14
    images = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.qint32)
    size = np.array([2, 2], dtype=np.int32)
    min_val = np.array(-10.0, dtype=np.float32)
    max_val = np.array(10.0, dtype=np.float32)
    align_corners = False
    half_pixel_centers = False

    input_dict = {
        "images": images,
        "size": size,
        "min": min_val,
        "max": max_val,
        "align_corners": align_corners,
        "half_pixel_centers": half_pixel_centers,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedResizeBilinear"] = tf_raw_ops_QuantizedResizeBilinear_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedResizeBilinear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedResizeBilinear'.")

check_valid('tf.raw_ops.QuantizedResizeBilinear', generated_inputs['tf.raw_ops.QuantizedResizeBilinear'], lib="tf", suffix=0)
