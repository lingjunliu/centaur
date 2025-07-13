
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
    image = np.random.rand(1, 10, 10, 3).astype(np.float32)
    boxes = np.array([[0.2, 0.2, 0.8, 0.8]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([5, 5], dtype=np.int32)
    method = "nearest"
    extrapolation_value = 1.0
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
    image = np.random.rand(2, 20, 20, 1).astype(np.float32)
    boxes = np.array([[0.1, 0.1, 0.9, 0.9], [0.3, 0.3, 0.7, 0.7]], dtype=np.float32)
    box_ind = np.array([0, 1], dtype=np.int32)
    crop_size = np.array([10, 10], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = -1.0
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

    # Input 4: boxes outside [0, 1]
    image = np.random.rand(1, 5, 5, 3).astype(np.float32)
    boxes = np.array([[-0.5, -0.5, 1.5, 1.5]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([3, 3], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 0.5
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

   # Input 5: uint8 image
    image = np.random.randint(0, 256, size=(1, 5, 5, 3), dtype=np.uint8)
    boxes = np.array([[0.1, 0.1, 0.9, 0.9]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([3, 3], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 0.0
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

    # Input 6:  boxes with y1 > y2 and x1 > x2
    image = np.random.rand(1, 5, 5, 3).astype(np.float32)
    boxes = np.array([[0.9, 0.9, 0.1, 0.1]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([3, 3], dtype=np.int32)
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

    # Input 7 : 2 images with different boxes
    image = np.random.rand(2, 10, 10, 3).astype(np.float32)
    boxes = np.array([[0.2, 0.2, 0.8, 0.8], [0.1, 0.1, 0.5, 0.5]], dtype=np.float32)
    box_ind = np.array([0, 1], dtype=np.int32)
    crop_size = np.array([5, 5], dtype=np.int32)
    method = "nearest"
    extrapolation_value = 0.0
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

    # Input 8: int32 image
    image = np.random.randint(0, 100, size=(1, 5, 5, 3), dtype=np.int32)
    boxes = np.array([[0.1, 0.1, 0.9, 0.9]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([3, 3], dtype=np.int32)
    method = "bilinear"
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

    # Input 9
    image = np.random.rand(1, 10, 10, 3).astype(np.float32)
    boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([10, 10], dtype=np.int32)
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

    # Input 10: crop_size larger than image
    image = np.random.rand(1, 5, 5, 3).astype(np.float32)
    boxes = np.array([[0.1, 0.1, 0.9, 0.9]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([10, 10], dtype=np.int32)
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

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_crop_and_resize_inputs()

for input_dict in inputs:
  for k, v in input_dict.items():
    if isinstance(v, np.ndarray):
      input_dict[k] = tf.convert_to_tensor(v)


generated_inputs["tf.raw_ops.CropAndResize"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.CropAndResize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.CropAndResize'.")

check_valid('tf.raw_ops.CropAndResize', generated_inputs['tf.raw_ops.CropAndResize'], lib="tf", suffix=0)
