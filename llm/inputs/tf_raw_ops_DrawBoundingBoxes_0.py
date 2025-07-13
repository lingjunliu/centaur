
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

    # Input 2: Multiple bounding boxes
    images = np.random.rand(1, 50, 100, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9], [0.6, 0.1, 0.8, 0.3]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple images in a batch
    images = np.random.rand(2, 64, 64, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]], [[0.2, 0.3, 0.6, 0.7]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different image dimensions
    images = np.random.rand(1, 256, 256, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Box at the edge of the image
    images = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Half type
    images = np.random.rand(1, 32, 32, 3).astype(np.float16)
    boxes = np.array([[[0.2, 0.2, 0.8, 0.8]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Name specified
    images = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.array([[[0.3, 0.3, 0.7, 0.7]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": "my_bounding_boxes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small values
    images = np.random.rand(1, 16, 16, 3).astype(np.float32)
    boxes = np.array([[[0.01, 0.02, 0.05, 0.09]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: boxes values close to 1
    images = np.random.rand(1, 16, 16, 3).astype(np.float32)
    boxes = np.array([[[0.91, 0.92, 0.95, 0.99]]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Batch size larger than 1, many boxes
    images = np.random.rand(3, 32, 32, 3).astype(np.float32)
    boxes = np.random.rand(3, 5, 4).astype(np.float32)
    boxes = np.clip(boxes, 0.0, 1.0) # ensure boxes are within [0, 1]
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
