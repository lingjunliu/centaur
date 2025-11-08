
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_draw_bounding_boxes_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with 4D image and 3D boxes
    images = np.random.rand(2, 10, 10, 3).astype(np.float32)
    boxes = np.random.rand(2, 3, 4).astype(np.float32)
    
    input_dict = {
        "name": "test1",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Different batch size
    images = np.random.rand(1, 5, 5, 3).astype(np.float32)
    boxes = np.random.rand(1, 2, 4).astype(np.float32)
    
    input_dict = {
        "name": "test2",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input  3: Different image size
    images = np.random.rand(1, 20, 30, 4).astype(np.float32)
    boxes = np.random.rand(1, 1, 4).astype(np.float32)
    
    input_dict = {
        "name": "test3",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Different number of bounding boxes
    images = np.random.rand(3, 15, 15, 3).astype(np.float32)
    boxes = np.random.rand(3, 5, 4).astype(np.float32)
    
    input_dict = {
        "name": "test4",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: With negative values in boxes (valid since API does not restrict negative values)
    images = np.random.rand(2, 10, 10, 3).astype(np.float32)
    boxes = np.random.rand(2, 3, 4).astype(np.float32) - 0.5
    
    input_dict = {
        "name": "test5",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Float values in boxes
    images = np.random.rand(1, 20, 20, 3).astype(np.float32)
    boxes = np.random.rand(1, 2, 4).astype(np.float32)
    
    input_dict = {
        "name": "test6",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Different depth
    images = np.random.rand(2, 10, 10, 1).astype(np.float32)
    boxes = np.random.rand(2, 2, 4).astype(np.float32)
    
    input_dict = {
        "name": "test7",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Different image dimensions (height/width)
    images = np.random.rand(1, 25, 30, 3).astype(np.float32)
    boxes = np.random.rand(1, 1, 4).astype(np.float32)
    
    input_dict = {
        "name": "test8",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: With multiple bounding boxes per image
    images = np.random.rand(3, 10, 10, 3).astype(np.float32)
    boxes = np.random.rand(3, 4, 4).astype(np.float32)
    
    input_dict = {
        "name": "test9",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Different image types (e.g. half)
    images = np.random.rand(2, 15, 15, 3).astype(np.float16)
    boxes = np.random.rand(2, 3, 4).astype(np.float32)
    
    input_dict = {
        "name": "test10",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.DrawBoundingBoxes"] = generate_draw_bounding_boxes_inputs()

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
