
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DrawBoundingBoxesV2_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    images = np.random.rand(1, 100, 200, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "colors": colors, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple boxes
    images = np.random.rand(1, 50, 50, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.9, 0.9]]]).astype(np.float32)
    colors = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "colors": colors, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple images
    images = np.random.rand(2, 64, 64, 3).astype(np.float32)
    boxes = np.array([[[0.2, 0.2, 0.4, 0.4]], [[0.7, 0.7, 0.9, 0.9]]]).astype(np.float32)
    colors = np.array([[1.0, 1.0, 0.0, 1.0]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "colors": colors, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Half type
    images = np.random.rand(1, 32, 32, 3).astype(np.float16)
    boxes = np.array([[[0.3, 0.3, 0.6, 0.6]]]).astype(np.float32)
    colors = np.array([[1.0, 0.5, 0.0, 1.0]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "colors": colors, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different color
    images = np.random.rand(1, 20, 20, 3).astype(np.float32)
    boxes = np.array([[[0.4, 0.4, 0.7, 0.7]]]).astype(np.float32)
    colors = np.array([[0.5, 0.5, 0.5, 0.5]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "colors": colors, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boxes covering entire image
    images = np.random.rand(1, 40, 40, 3).astype(np.float32)
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]]).astype(np.float32)
    colors = np.array([[0.0, 1.0, 1.0, 1.0]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "colors": colors, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different batch size, different number of boxes
    images = np.random.rand(3, 16, 16, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.1, 0.9, 0.9], [0.2, 0.3, 0.5, 0.6]], [[0.3, 0.4, 0.7, 0.8], [0.1, 0.5, 0.6, 0.9]], [[0.6, 0.2, 0.8, 0.4], [0.7, 0.3, 0.9, 0.5]]]).astype(np.float32)
    colors = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "colors": colors, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single channel image
    images = np.random.rand(1, 100, 100, 1).astype(np.float32)
    boxes = np.array([[[0.2, 0.3, 0.6, 0.7]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "colors": colors, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty boxes tensor
    images = np.random.rand(1, 50, 50, 3).astype(np.float32)
    boxes = np.array([[]]).reshape(1, 0, 4).astype(np.float32)
    colors = np.array([[1.0, 1.0, 1.0, 1.0]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "colors": colors, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: name parameter given
    images = np.random.rand(1, 80, 80, 3).astype(np.float32)
    boxes = np.array([[[0.0, 0.0, 0.5, 0.5]]]).astype(np.float32)
    colors = np.array([[1.0, 0.0, 1.0, 1.0]]).astype(np.float32)
    input_dict = {"images": images, "boxes": boxes, "colors": colors, "name": "my_bounding_boxes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DrawBoundingBoxesV2"] = tf_raw_ops_DrawBoundingBoxesV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DrawBoundingBoxesV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DrawBoundingBoxesV2'.")

check_valid('tf.raw_ops.DrawBoundingBoxesV2', generated_inputs['tf.raw_ops.DrawBoundingBoxesV2'], lib="tf", suffix=0)
