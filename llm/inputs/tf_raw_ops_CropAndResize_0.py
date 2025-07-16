
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_crop_and_resize_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.float32)
    boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([2, 2], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 0.0
    name = None

    input_dict = {
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "crop_size": crop_size,
        "method": method,
        "extrapolation_value": extrapolation_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    boxes = np.array([[0, 0, 0.5, 0.5]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([3, 3], dtype=np.int32)
    method = "nearest"
    extrapolation_value = 1.0
    name = None

    input_dict = {
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "crop_size": crop_size,
        "method": method,
        "extrapolation_value": extrapolation_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float32)
    boxes = np.array([[0, 0, 1, 1], [0, 0, 0.5, 0.5]], dtype=np.float32)
    box_ind = np.array([0, 1], dtype=np.int32)
    crop_size = np.array([4, 4], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = -1.0
    name = None

    input_dict = {
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "crop_size": crop_size,
        "method": method,
        "extrapolation_value": extrapolation_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.float32)
    boxes = np.array([[0.1, 0.2, 0.8, 0.9]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([5, 5], dtype=np.int32)
    method = "nearest"
    extrapolation_value = 2.0
    name = None

    input_dict = {
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "crop_size": crop_size,
        "method": method,
        "extrapolation_value": extrapolation_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    image = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.uint8)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([2, 2], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 0.0
    name = None

    input_dict = {
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "crop_size": crop_size,
        "method": method,
        "extrapolation_value": extrapolation_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    image = np.array([[[[1, 2, 3]]]], dtype=np.float64)
    boxes = np.array([[0.0, 0.0, 0.5, 0.5]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([1, 1], dtype=np.int32)
    method = "nearest"
    extrapolation_value = 0.0
    name = None

    input_dict = {
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "crop_size": crop_size,
        "method": method,
        "extrapolation_value": extrapolation_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    image = np.array([[[[1, 2], [3, 4]]]], dtype=np.int32)
    boxes = np.array([[0.0, 0.0, 1.5, 1.5]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([3, 3], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 1.0
    name = None

    input_dict = {
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "crop_size": crop_size,
        "method": method,
        "extrapolation_value": extrapolation_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    image = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    boxes = np.array([[0.8, 0.8, 0.2, 0.2]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([2, 2], dtype=np.int32)
    method = "nearest"
    extrapolation_value = 0.5
    name = None

    input_dict = {
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "crop_size": crop_size,
        "method": method,
        "extrapolation_value": extrapolation_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
   # Input 9
    image = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.float32)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([1, 1], dtype=np.int32)
    method = "nearest"
    extrapolation_value = 0.0
    name = None

    input_dict = {
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "crop_size": crop_size,
        "method": method,
        "extrapolation_value": extrapolation_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    image = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.float32)
    boxes = np.array([[0, 0, 1, 1], [0.5, 0.5, 0.75, 0.75]], dtype=np.float32)
    box_ind = np.array([0, 0], dtype=np.int32)
    crop_size = np.array([2, 2], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 0.0
    name = None

    input_dict = {
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "crop_size": crop_size,
        "method": method,
        "extrapolation_value": extrapolation_value,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.CropAndResize"] = tf_raw_ops_crop_and_resize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.CropAndResize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.CropAndResize'.")

check_valid('tf.raw_ops.CropAndResize', generated_inputs['tf.raw_ops.CropAndResize'], lib="tf", suffix=0)
