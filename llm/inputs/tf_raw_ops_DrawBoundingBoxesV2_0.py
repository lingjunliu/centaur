
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_draw_bounding_boxes_v2_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    images = np.random.rand(1, 100, 200, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]]).astype(np.float32)

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple boxes
    images = np.random.rand(1, 100, 200, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9], [0.6, 0.1, 0.8, 0.3]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]).astype(np.float32)

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple images
    images = np.random.rand(2, 100, 200, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]], [[0.6, 0.1, 0.8, 0.3]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]]).astype(np.float32)

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different image size
    images = np.random.rand(1, 50, 100, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]]).astype(np.float32)

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different color
    images = np.random.rand(1, 100, 200, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]]).astype(np.float32)
    colors = np.array([[0.0, 1.0, 0.0, 1.0]]).astype(np.float32)

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: box near the edge
    images = np.random.rand(1, 100, 200, 3).astype(np.float32)
    boxes = np.array([[[0.01, 0.02, 0.98, 0.99]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]]).astype(np.float32)

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: half type
    images = np.random.rand(1, 100, 200, 3).astype(np.float16)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]]).astype(np.float32)

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero box
    images = np.random.rand(1, 100, 200, 3).astype(np.float32)
    boxes = np.array([[[0.0, 0.0, 0.0, 0.0]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]]).astype(np.float32)

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2 colors 2 images
    images = np.random.rand(2, 100, 200, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]], [[0.1, 0.2, 0.5, 0.9]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]).astype(np.float32)

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: different depth
    images = np.random.rand(1, 100, 200, 1).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]]).astype(np.float32)

    input_dict = {
        "images": images,
        "boxes": boxes,
        "colors": colors,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DrawBoundingBoxesV2"] = tf_raw_ops_draw_bounding_boxes_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DrawBoundingBoxesV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DrawBoundingBoxesV2'.")

check_valid('tf.raw_ops.DrawBoundingBoxesV2', generated_inputs['tf.raw_ops.DrawBoundingBoxesV2'], lib="tf", suffix=0)
