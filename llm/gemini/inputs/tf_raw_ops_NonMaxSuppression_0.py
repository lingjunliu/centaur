
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_NonMaxSuppression_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = 0.5
    name = None

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different iou_threshold
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = 0.9
    name = None
    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger max_output_size
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = np.array(5, dtype=np.int32)
    iou_threshold = 0.5
    name = None
    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Overlapping boxes
    boxes = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.array(3, dtype=np.int32)
    iou_threshold = 0.5
    name = None
    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Non-overlapping boxes
    boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.array(3, dtype=np.int32)
    iou_threshold = 0.5
    name = None
    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: One box
    boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    max_output_size = np.array(1, dtype=np.int32)
    iou_threshold = 0.5
    name = None
    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero max_output_size
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75], dtype=np.float32)
    max_output_size = np.array(0, dtype=np.int32)
    iou_threshold = 0.5
    name = None
    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Larger coordinates
    boxes = np.array([[100, 100, 200, 200], [150, 150, 250, 250]], dtype=np.float32)
    scores = np.array([0.9, 0.75], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = 0.5
    name = None
    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: boxes with same values
    boxes = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
    scores = np.array([0.9, 0.75], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = 0.5
    name = None
    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different values for boxes
    boxes = np.array([[0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = 0.5
    name = None

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.NonMaxSuppression"] = tf_raw_ops_NonMaxSuppression_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NonMaxSuppression' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NonMaxSuppression'.")

check_valid('tf.raw_ops.NonMaxSuppression', generated_inputs['tf.raw_ops.NonMaxSuppression'], lib="tf", suffix=0)
