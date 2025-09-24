
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_crop_and_resize_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]], dtype=np.float32)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([2, 2], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 0.0
    name = "crop_and_resize_1"

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
    image = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    boxes = np.array([[0.0, 0.0, 0.5, 0.5]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([1, 1], dtype=np.int32)
    method = "nearest"
    extrapolation_value = 0.0
    name = "crop_and_resize_2"

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
    image = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    boxes = np.array([[0.5, 0.5, 1.0, 1.0]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([1, 1], dtype=np.int32)
    method = "nearest"
    extrapolation_value = 0.0
    name = "crop_and_resize_3"

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
    image = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    boxes = np.array([[0.0, 0.0, 1.5, 1.5]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([3, 3], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 1.0
    name = "crop_and_resize_4"

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
    image = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    boxes = np.array([[-0.5, -0.5, 0.5, 0.5]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([3, 3], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 1.0
    name = "crop_and_resize_5"

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
    image = np.random.rand(2, 32, 32, 3).astype(np.float32)
    boxes = np.random.rand(5, 4).astype(np.float32)
    box_ind = np.random.randint(0, 2, size=(5,)).astype(np.int32)
    crop_size = np.array([16, 16], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 0.0
    name = "crop_and_resize_6"

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

   # Input 7: Different extrapolation value
    image = np.random.rand(1, 28, 28, 1).astype(np.float32)
    boxes = np.array([[0.1, 0.2, 0.8, 0.9]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([24, 24], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = -1.0
    name = "crop_and_resize_7"

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

    # Input 8: Nearest neighbor, different crop size
    image = np.random.rand(1, 64, 64, 3).astype(np.float32)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([32, 32], dtype=np.int32)
    method = "nearest"
    extrapolation_value = 0.0
    name = "crop_and_resize_8"

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

    # Input 9: Multiple boxes
    image = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]], dtype=np.float32)
    box_ind = np.array([0, 0], dtype=np.int32)
    crop_size = np.array([16, 16], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 0.0
    name = "crop_and_resize_9"

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

   # Input 10: Multiple batches
    image = np.random.rand(2, 32, 32, 3).astype(np.float32)
    boxes = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]], dtype=np.float32)
    box_ind = np.array([0, 1], dtype=np.int32)
    crop_size = np.array([16, 16], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 0.0
    name = "crop_and_resize_10"

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

    # Input 11: y1 > y2
    image = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.array([[0.8, 0.0, 0.2, 1.0]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([16, 16], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 0.0
    name = "crop_and_resize_11"

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

    # Input 12: x1 > x2
    image = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.array([[0.0, 0.8, 1.0, 0.2]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([16, 16], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 0.0
    name = "crop_and_resize_12"

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
