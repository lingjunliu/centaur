
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_CropAndResizeGradBoxes_inputs():
    list_of_inputs = []

    # Input 1
    grads = np.random.rand(2, 10, 10, 3).astype(np.float32)
    image = np.random.rand(1, 20, 20, 3).astype(np.float32)
    boxes = np.array([[0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8]], dtype=np.float32)
    box_ind = np.array([0, 0], dtype=np.int32)
    method = "bilinear"
    name = "crop_and_resize_grad_boxes_1"

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    grads = np.random.rand(1, 5, 5, 1).astype(np.float32)
    image = np.random.rand(2, 15, 15, 1).astype(np.float32)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    method = "bilinear"
    name = "crop_and_resize_grad_boxes_2"

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    grads = np.random.rand(3, 7, 7, 2).astype(np.float32)
    image = np.random.rand(3, 14, 14, 2).astype(np.float32)
    boxes = np.array([[0.2, 0.3, 0.7, 0.8], [0.1, 0.0, 0.9, 0.5], [0.5, 0.5, 0.6, 0.6]], dtype=np.float32)
    box_ind = np.array([0, 1, 2], dtype=np.int32)
    method = "bilinear"
    name = "crop_and_resize_grad_boxes_3"

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    grads = np.random.rand(4, 3, 3, 4).astype(np.float32)
    image = np.random.rand(4, 6, 6, 4).astype(np.float32)
    boxes = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.8, 0.8], [0.3, 0.3, 0.7, 0.7]], dtype=np.float32)
    box_ind = np.array([0, 1, 2, 3], dtype=np.int32)
    method = "bilinear"
    name = "crop_and_resize_grad_boxes_4"

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    grads = np.random.rand(1, 12, 12, 3).astype(np.float32)
    image = np.random.rand(1, 24, 24, 3).astype(np.float32)
    boxes = np.array([[0.8, 0.8, 0.2, 0.2]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    method = "bilinear"
    name = "crop_and_resize_grad_boxes_5"

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    grads = np.random.rand(2, 8, 8, 1).astype(np.float32)
    image = np.random.rand(2, 16, 16, 1).astype(np.float32)
    boxes = np.array([[0.1, 0.9, 0.9, 0.1], [0.9, 0.1, 0.1, 0.9]], dtype=np.float32)
    box_ind = np.array([0, 1], dtype=np.int32)
    method = "bilinear"
    name = "crop_and_resize_grad_boxes_6"

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    grads = np.zeros((1, 1, 1, 1), dtype=np.float32)
    image = np.ones((1, 2, 2, 1), dtype=np.float32)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    method = "bilinear"
    name = "crop_and_resize_grad_boxes_7"

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    grads = np.random.rand(1, 1, 1, 3).astype(np.float32)
    image = np.random.rand(1, 2, 2, 3).astype(np.float32)
    boxes = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    method = "bilinear"
    name = "crop_and_resize_grad_boxes_8"

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    grads = np.random.rand(2, 4, 4, 1).astype(np.float32)
    image = np.random.rand(3, 8, 8, 1).astype(np.float32)
    boxes = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    box_ind = np.array([0, 2], dtype=np.int32)
    method = "bilinear"
    name = "crop_and_resize_grad_boxes_9"

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    grads = np.random.rand(1, 2, 2, 1).astype(np.float32)
    image = np.random.rand(1, 4, 4, 1).astype(np.float32)
    boxes = np.array([[1.1, 1.1, -0.1, -0.1]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    method = "bilinear"
    name = "crop_and_resize_grad_boxes_10"

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.CropAndResizeGradBoxes"] = tf_raw_ops_CropAndResizeGradBoxes_inputs()
api = tf.raw_ops.CropAndResizeGradBoxes
for i in range(len(generated_inputs["tf.raw_ops.CropAndResizeGradBoxes"])):
    inp = generated_inputs["tf.raw_ops.CropAndResizeGradBoxes"][i]
    generated_inputs["tf.raw_ops.CropAndResizeGradBoxes"][i] = {"args": [], "kwargs": inp}

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.CropAndResizeGradBoxes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.CropAndResizeGradBoxes'.")

check_valid('tf.raw_ops.CropAndResizeGradBoxes', generated_inputs['tf.raw_ops.CropAndResizeGradBoxes'], lib="tf", suffix=0)
