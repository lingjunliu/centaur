
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_draw_bounding_boxes_v2_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    images = np.random.rand(1, 100, 200, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]]).astype(np.float32)
    name = None

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple bounding boxes
    images = np.random.rand(1, 50, 50, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.9, 0.9]]]).astype(np.float32)
    colors = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]).astype(np.float32)
    name = "multiple_boxes"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple images in a batch
    images = np.random.rand(2, 64, 64, 3).astype(np.float32)
    boxes = np.array([[[0.2, 0.3, 0.7, 0.8]], [[0.1, 0.4, 0.6, 0.9]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 1.0, 0.5]]).astype(np.float32)
    name = "multiple_images"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different image dimensions, half type
    images = np.random.rand(1, 32, 128, 3).astype(np.float16)
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]]).astype(np.float32)
    colors = np.array([[0.5, 0.5, 0.5, 0.5]]).astype(np.float32)
    name = "half_type"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boxes outside image bounds (still valid, parts are clipped)
    images = np.random.rand(1, 20, 20, 3).astype(np.float32)
    boxes = np.array([[[ -0.1, -0.1, 1.2, 1.2]]]).astype(np.float32)
    colors = np.array([[0.0, 1.0, 1.0, 1.0]]).astype(np.float32)
    name = "outside_bounds"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty boxes
    images = np.random.rand(1, 25, 25, 3).astype(np.float32)
    boxes = np.array([[]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]]).astype(np.float32)
    name = "empty_boxes"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: More boxes than colors (cycles through colors)
    images = np.random.rand(1, 40, 40, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.0, 0.1, 0.2, 0.3]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]).astype(np.float32)
    name = "more_boxes_than_colors"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single pixel image
    images = np.random.rand(1, 1, 1, 3).astype(np.float32)
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]]).astype(np.float32)
    colors = np.array([[1.0, 1.0, 1.0, 1.0]]).astype(np.float32)
    name = "single_pixel"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Batch size greater than 1, more than one color
    images = np.random.rand(2, 20, 20, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.8]], [[0.3, 0.4, 0.7, 0.9]]]).astype(np.float32)
    colors = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]]).astype(np.float32)
    name = "batch_size_colors"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different Box Coordinates
    images = np.random.rand(1, 30, 30, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.1, 0.1, 0.1]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]]).astype(np.float32)
    name = "different_box"

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_draw_bounding_boxes_v2_inputs()
generated_inputs["tf.raw_ops.DrawBoundingBoxesV2"] = []
for input_dict in inputs:
  generated_inputs["tf.raw_ops.DrawBoundingBoxesV2"].append({
      "images": input_dict["images"],
      "boxes": input_dict["boxes"],
      "colors": input_dict["colors"],
      "name": input_dict["name"]
  })

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DrawBoundingBoxesV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DrawBoundingBoxesV2'.")

check_valid('tf.raw_ops.DrawBoundingBoxesV2', generated_inputs['tf.raw_ops.DrawBoundingBoxesV2'], lib="tf", suffix=0)
