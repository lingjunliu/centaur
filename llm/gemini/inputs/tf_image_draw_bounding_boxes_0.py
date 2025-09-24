
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_draw_bounding_boxes_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    images = np.zeros((1, 100, 200, 3), dtype=np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    name = "basic_case"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple boxes
    images = np.zeros((1, 50, 50, 3), dtype=np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9], [0.6, 0.1, 0.8, 0.4]]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    name = "multiple_boxes"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple images in batch
    images = np.zeros((2, 30, 40, 3), dtype=np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]], [[0.2, 0.3, 0.6, 0.8]]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    name = "multiple_images"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4: Different image size
    images = np.zeros((1, 256, 256, 3), dtype=np.float32)
    boxes = np.array([[[0.05, 0.15, 0.95, 0.85]]], dtype=np.float32)
    colors = np.array([[0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    name = "different_image_size"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single channel image
    images = np.zeros((1, 64, 64, 1), dtype=np.float32)
    boxes = np.array([[[0.2, 0.3, 0.7, 0.6]]], dtype=np.float32)
    colors = np.array([[0.5, 0.5, 0.5, 1.0]], dtype=np.float32)
    name = "single_channel"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float16 images
    images = np.zeros((1, 100, 200, 3), dtype=np.float16)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    name = "float16_images"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero size boxes
    images = np.zeros((1, 100, 200, 3), dtype=np.float32)
    boxes = np.array([[[0.1, 0.1, 0.1, 0.1]]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    name = "zero_size_boxes"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Boxes near edges
    images = np.zeros((1, 100, 200, 3), dtype=np.float32)
    boxes = np.array([[[0.01, 0.02, 0.98, 0.99]]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    name = "boxes_near_edges"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: One image, two colors, one box
    images = np.zeros((1, 3, 3, 3), dtype=np.float32)
    boxes = np.array([[[0, 0, 1, 1]]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    name = "one_image_two_colors"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multiple boxes and different colors
    images = np.zeros((1, 64, 64, 3), dtype=np.float32)
    boxes = np.array([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    name = "multiple_boxes_and_colors"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.draw_bounding_boxes"] = tf_image_draw_bounding_boxes_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.draw_bounding_boxes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.draw_bounding_boxes'.")

check_valid('tf.image.draw_bounding_boxes', generated_inputs['tf.image.draw_bounding_boxes'], lib="tf", suffix=0)
