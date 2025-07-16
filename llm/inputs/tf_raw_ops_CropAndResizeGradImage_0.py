
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_crop_and_resize_grad_image_inputs():
    list_of_inputs = []

    # Input 1
    grads = np.random.rand(1, 8, 8, 3).astype(np.float32)
    boxes = np.array([[0.1, 0.1, 0.9, 0.9]]).astype(np.float32)
    box_ind = np.array([0]).astype(np.int32)
    image_size = np.array([1, 64, 64, 3]).astype(np.int32)
    T = tf.float32
    method = "bilinear"
    name = "crop_and_resize_grad_image_1"

    input_dict = {
        "grads": grads,
        "boxes": boxes,
        "box_ind": box_ind,
        "image_size": image_size,
        "T": T,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    grads = np.random.rand(2, 16, 16, 1).astype(np.float32)
    boxes = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]).astype(np.float32)
    box_ind = np.array([0, 0]).astype(np.int32)
    image_size = np.array([1, 32, 32, 1]).astype(np.int32)
    T = tf.float32
    method = "bilinear"
    name = "crop_and_resize_grad_image_2"

    input_dict = {
        "grads": grads,
        "boxes": boxes,
        "box_ind": box_ind,
        "image_size": image_size,
        "T": T,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    grads = np.random.rand(3, 4, 4, 3).astype(np.float32)
    boxes = np.array([[0.2, 0.2, 0.8, 0.8], [0.1, 0.3, 0.7, 0.9], [0.0, 0.0, 1.0, 1.0]]).astype(np.float32)
    box_ind = np.array([0, 1, 0]).astype(np.int32)
    image_size = np.array([2, 16, 16, 3]).astype(np.int32)
    T = tf.float32
    method = "bilinear"
    name = "crop_and_resize_grad_image_3"

    input_dict = {
        "grads": grads,
        "boxes": boxes,
        "box_ind": box_ind,
        "image_size": image_size,
        "T": T,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    grads = np.random.rand(1, 8, 8, 3).astype(np.float32)
    boxes = np.array([[0.1, 0.9, 0.9, 0.1]]).astype(np.float32)
    box_ind = np.array([0]).astype(np.int32)
    image_size = np.array([1, 64, 64, 3]).astype(np.int32)
    T = tf.float32
    method = "bilinear"
    name = "crop_and_resize_grad_image_4"

    input_dict = {
        "grads": grads,
        "boxes": boxes,
        "box_ind": box_ind,
        "image_size": image_size,
        "T": T,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.CropAndResizeGradImage"] = tf_raw_ops_crop_and_resize_grad_image_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.CropAndResizeGradImage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.CropAndResizeGradImage'.")

check_valid('tf.raw_ops.CropAndResizeGradImage', generated_inputs['tf.raw_ops.CropAndResizeGradImage'], lib="tf", suffix=0)
