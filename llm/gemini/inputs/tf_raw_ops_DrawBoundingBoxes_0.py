
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_draw_bounding_boxes_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    images = np.random.rand(1, 100, 200, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple boxes, single image
    images = np.random.rand(1, 100, 200, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9], [0.6, 0.1, 0.8, 0.3]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple images, multiple boxes
    images = np.random.rand(2, 100, 200, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9], [0.6, 0.1, 0.8, 0.3]], [[0.2, 0.3, 0.7, 0.8], [0.0, 0.5, 0.4, 0.6]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different image size
    images = np.random.rand(1, 50, 80, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Box at the edge
    images = np.random.rand(1, 100, 200, 3).astype(np.float32)
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty boxes
    images = np.random.rand(1, 100, 200, 3).astype(np.float32)
    boxes = np.array([[]]).reshape(1,0,4).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Half type images
    images = np.random.rand(1, 100, 200, 3).astype(np.float16)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multiple half type images and boxes
    images = np.random.rand(2, 100, 200, 3).astype(np.float16)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9], [0.6, 0.1, 0.8, 0.3]], [[0.2, 0.3, 0.7, 0.8], [0.0, 0.5, 0.4, 0.6]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large image
    images = np.random.rand(1, 512, 512, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Very small image
    images = np.random.rand(1, 10, 10, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DrawBoundingBoxes"] = tf_raw_ops_draw_bounding_boxes_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DrawBoundingBoxes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DrawBoundingBoxes'.")

check_valid('tf.raw_ops.DrawBoundingBoxes', generated_inputs['tf.raw_ops.DrawBoundingBoxes'], lib="tf", suffix=0)
