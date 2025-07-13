
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_crop_and_resize_grad_boxes_inputs():
    list_of_inputs = []

    # Input 1
    grads = np.random.rand(1, 24, 24, 3).astype(np.float32)
    image = np.random.rand(1, 100, 100, 3).astype(np.float32)
    boxes = np.array([[0.1, 0.1, 0.9, 0.9]]).astype(np.float32)
    box_ind = np.array([0]).astype(np.int32)
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
    grads = np.random.rand(2, 12, 12, 1).astype(np.float32)
    image = np.random.rand(2, 50, 50, 1).astype(np.float32)
    boxes = np.array([[0.2, 0.2, 0.8, 0.8], [0.3, 0.3, 0.7, 0.7]]).astype(np.float32)
    box_ind = np.array([0, 1]).astype(np.int32)
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
    grads = np.random.rand(3, 8, 8, 3).astype(np.float32)
    image = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8]]).astype(np.float32)
    box_ind = np.array([0, 0, 0]).astype(np.int32)
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
    grads = np.random.rand(1, 16, 16, 3).astype(np.float32)
    image = np.random.rand(1, 64, 64, 3).astype(np.float32)
    boxes = np.array([[0.1, 0.2, 0.7, 0.8]]).astype(np.float32)
    box_ind = np.array([0]).astype(np.int32)
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
    grads = np.random.rand(4, 4, 4, 1).astype(np.float32)
    image = np.random.rand(4, 16, 16, 1).astype(np.float32)
    boxes = np.array([[0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75], [0.5, 0.5, 1.0, 1.0], [0.1, 0.3, 0.6, 0.8]]).astype(np.float32)
    box_ind = np.array([0, 1, 2, 3]).astype(np.int32)
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

    # Input 6: y1 > y2, x1 > x2
    grads = np.random.rand(1, 8, 8, 3).astype(np.float32)
    image = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.array([[0.8, 0.7, 0.2, 0.1]]).astype(np.float32)
    box_ind = np.array([0]).astype(np.int32)
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

    # Input 7: boxes outside [0, 1] range
    grads = np.random.rand(1, 8, 8, 3).astype(np.float32)
    image = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.array([[-0.1, -0.2, 1.1, 1.2]]).astype(np.float32)
    box_ind = np.array([0]).astype(np.int32)
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

    # Input 8: multiple boxes, same image
    grads = np.random.rand(2, 8, 8, 3).astype(np.float32)
    image = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.array([[0.1, 0.1, 0.4, 0.4], [0.6, 0.6, 0.9, 0.9]]).astype(np.float32)
    box_ind = np.array([0, 0]).astype(np.int32)
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

    # Input 9: different image types (uint8)
    grads = np.random.rand(1, 8, 8, 3).astype(np.float32)
    image = np.random.randint(0, 256, size=(1, 32, 32, 3)).astype(np.uint8)
    boxes = np.array([[0.1, 0.1, 0.9, 0.9]]).astype(np.float32)
    box_ind = np.array([0]).astype(np.int32)
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

    # Input 10: 2 batches, different boxes
    grads = np.random.rand(2, 8, 8, 3).astype(np.float32)
    image = np.random.rand(2, 32, 32, 3).astype(np.float32)
    boxes = np.array([[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.8, 0.8]]).astype(np.float32)
    box_ind = np.array([0, 1]).astype(np.int32)
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
inputs = tf_raw_ops_crop_and_resize_grad_boxes_inputs()
generated_inputs["tf.raw_ops.CropAndResizeGradBoxes"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.CropAndResizeGradBoxes"].append({k: input_dict[k] for k in ("grads", "image", "boxes", "box_ind", "method", "name")})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.CropAndResizeGradBoxes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.CropAndResizeGradBoxes'.")

check_valid('tf.raw_ops.CropAndResizeGradBoxes', generated_inputs['tf.raw_ops.CropAndResizeGradBoxes'], lib="tf", suffix=0)
