
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_crop_and_resize_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    image = np.random.rand(1, 256, 256, 3).astype(np.float32)
    boxes = np.array([[0.25, 0.25, 0.75, 0.75]], dtype=np.float32)
    box_indices = np.array([0], dtype=np.int32)
    crop_size = np.array([64, 64], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 0.0
    name = "crop_and_resize_1"
    input_dict = {"image": image, "boxes": boxes, "box_indices": box_indices, "crop_size": crop_size, "method": method, "extrapolation_value": extrapolation_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple boxes
    image = np.random.rand(1, 128, 128, 1).astype(np.float32)
    boxes = np.array([[0.1, 0.1, 0.4, 0.4], [0.6, 0.6, 0.9, 0.9]], dtype=np.float32)
    box_indices = np.array([0, 0], dtype=np.int32)
    crop_size = np.array([32, 32], dtype=np.int32)
    method = "nearest"
    extrapolation_value = 1.0
    name = "crop_and_resize_2"
    input_dict = {"image": image, "boxes": boxes, "box_indices": box_indices, "crop_size": crop_size, "method": method, "extrapolation_value": extrapolation_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different batch index
    image = np.random.rand(2, 64, 64, 3).astype(np.float32)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    box_indices = np.array([1], dtype=np.int32)
    crop_size = np.array([16, 16], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = -1.0
    name = "crop_and_resize_3"
    input_dict = {"image": image, "boxes": boxes, "box_indices": box_indices, "crop_size": crop_size, "method": method, "extrapolation_value": extrapolation_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Out of bounds boxes
    image = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.array([[-0.5, -0.5, 1.5, 1.5]], dtype=np.float32)
    box_indices = np.array([0], dtype=np.int32)
    crop_size = np.array([8, 8], dtype=np.int32)
    method = "nearest"
    extrapolation_value = 2.0
    name = "crop_and_resize_4"
    input_dict = {"image": image, "boxes": boxes, "box_indices": box_indices, "crop_size": crop_size, "method": method, "extrapolation_value": extrapolation_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Flipped boxes
    image = np.random.rand(1, 64, 64, 3).astype(np.float32)
    boxes = np.array([[0.75, 0.75, 0.25, 0.25]], dtype=np.float32)
    box_indices = np.array([0], dtype=np.int32)
    crop_size = np.array([16, 16], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 0.5
    name = "crop_and_resize_5"
    input_dict = {"image": image, "boxes": boxes, "box_indices": box_indices, "crop_size": crop_size, "method": method, "extrapolation_value": extrapolation_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different image size and crop size
    image = np.random.rand(1, 100, 150, 3).astype(np.float32)
    boxes = np.array([[0.1, 0.2, 0.8, 0.9]], dtype=np.float32)
    box_indices = np.array([0], dtype=np.int32)
    crop_size = np.array([30, 50], dtype=np.int32)
    method = "nearest"
    extrapolation_value = -0.5
    name = "crop_and_resize_6"
    input_dict = {"image": image, "boxes": boxes, "box_indices": box_indices, "crop_size": crop_size, "method": method, "extrapolation_value": extrapolation_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Batch size > 1, multiple boxes, mixed indices
    image = np.random.rand(3, 64, 64, 3).astype(np.float32)
    boxes = np.array([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7], [0.2, 0.2, 0.4, 0.4]], dtype=np.float32)
    box_indices = np.array([0, 1, 2], dtype=np.int32)
    crop_size = np.array([20, 20], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 0.0
    name = "crop_and_resize_7"
    input_dict = {"image": image, "boxes": boxes, "box_indices": box_indices, "crop_size": crop_size, "method": method, "extrapolation_value": extrapolation_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero sized crop
    image = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    box_indices = np.array([0], dtype=np.int32)
    crop_size = np.array([10, 10], dtype=np.int32)
    method = "nearest"
    extrapolation_value = 0.0
    name = "crop_and_resize_8"
    input_dict = {"image": image, "boxes": boxes, "box_indices": box_indices, "crop_size": crop_size, "method": method, "extrapolation_value": extrapolation_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different number of channels
    image = np.random.rand(1, 64, 64, 5).astype(np.float32)
    boxes = np.array([[0.25, 0.25, 0.75, 0.75]], dtype=np.float32)
    box_indices = np.array([0], dtype=np.int32)
    crop_size = np.array([16, 16], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 0.0
    name = "crop_and_resize_9"
    input_dict = {"image": image, "boxes": boxes, "box_indices": box_indices, "crop_size": crop_size, "method": method, "extrapolation_value": extrapolation_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Extrapolation value test
    image = np.zeros((1, 10, 10, 1), dtype=np.float32)
    boxes = np.array([[-0.5, -0.5, 1.5, 1.5]], dtype=np.float32)
    box_indices = np.array([0], dtype=np.int32)
    crop_size = np.array([5, 5], dtype=np.int32)
    method = "bilinear"
    extrapolation_value = 5.0
    name = "crop_and_resize_10"
    input_dict = {"image": image, "boxes": boxes, "box_indices": box_indices, "crop_size": crop_size, "method": method, "extrapolation_value": extrapolation_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.crop_and_resize"] = tf_image_crop_and_resize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.crop_and_resize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.crop_and_resize'.")

check_valid('tf.image.crop_and_resize', generated_inputs['tf.image.crop_and_resize'], lib="tf", suffix=0)
