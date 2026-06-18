
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def draw_bounding_boxes_inputs():
    list_of_inputs = []

    # Input 1: Basic Float32 image, 1 box
    images_1 = np.ones((1, 10, 10, 3), dtype=np.float32)
    boxes_1 = np.array([[[0.1, 0.1, 0.9, 0.9]]], dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_1",
        "images": images_1,
        "boxes": boxes_1
    })

    # Input 2: Float16 (half) image, 2 boxes, batch of 2
    images_2 = np.zeros((2, 5, 5, 1), dtype=np.float16)
    boxes_2 = np.array([
        [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]],
        [[0.2, 0.2, 0.8, 0.8], [0.1, 0.1, 0.9, 0.9]]
    ], dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_2",
        "images": images_2,
        "boxes": boxes_2
    })

    # Input 3: Zero boxes
    images_3 = np.random.rand(1, 20, 20, 4).astype(np.float32)
    boxes_3 = np.empty((1, 0, 4), dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_empty_boxes",
        "images": images_3,
        "boxes": boxes_3
    })

    # Input 4: Boxes out of bounds (negative/larger than 1)
    images_4 = np.zeros((3, 8, 8, 3), dtype=np.float32)
    boxes_4 = np.array([
        [[-0.2, -0.2, 1.2, 1.2]],
        [[-0.5, 0.0, 1.5, 1.0]],
        [[0.0, -0.5, 1.0, 1.5]]
    ], dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_out_of_bounds",
        "images": images_4,
        "boxes": boxes_4
    })

    # Input 5: Float16 image, multiple boxes
    images_5 = np.ones((1, 100, 100, 3), dtype=np.float16)
    boxes_5 = np.random.rand(1, 5, 4).astype(np.float32)
    list_of_inputs.append({
        "name": "draw_multiple_boxes",
        "images": images_5,
        "boxes": boxes_5
    })

    # Input 6: Zero size boxes (all 0.0)
    images_6 = np.random.rand(4, 4, 4, 3).astype(np.float32)
    boxes_6 = np.zeros((4, 1, 4), dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_zero_boxes",
        "images": images_6,
        "boxes": boxes_6
    })

    # Input 7: Ones boxes (all 1.0) with valid channel depth (3)
    images_7 = np.random.rand(1, 10, 10, 3).astype(np.float32)
    boxes_7 = np.ones((1, 2, 4), dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_ones_boxes",
        "images": images_7,
        "boxes": boxes_7
    })

    # Input 8: Float16 image, completely negative coords
    images_8 = np.ones((2, 15, 15, 3), dtype=np.float16)
    boxes_8 = np.array([
        [[-1.0, -1.0, -0.1, -0.1], [-2.0, -2.0, -0.5, -0.5], [-3.0, -3.0, -0.9, -0.9]],
        [[-1.0, -1.0, -0.1, -0.1], [-2.0, -2.0, -0.5, -0.5], [-3.0, -3.0, -0.9, -0.9]]
    ], dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_negative_coords",
        "images": images_8,
        "boxes": boxes_8
    })

    # Input 9: Minimal image (1x1)
    images_9 = np.ones((1, 1, 1, 3), dtype=np.float32)
    boxes_9 = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_minimal_image",
        "images": images_9,
        "boxes": boxes_9
    })

    # Input 10: Large batch and multiple boxes
    images_10 = np.random.rand(5, 10, 20, 3).astype(np.float32)
    boxes_10 = np.random.rand(5, 4, 4).astype(np.float32)
    list_of_inputs.append({
        "name": "draw_large_batch",
        "images": images_10,
        "boxes": boxes_10
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.DrawBoundingBoxes"] = draw_bounding_boxes_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DrawBoundingBoxes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DrawBoundingBoxes'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DrawBoundingBoxes', generated_inputs['tf.raw_ops.DrawBoundingBoxes'], lib="tf", suffix=0)
