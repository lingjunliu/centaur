
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_non_max_suppression_inputs():
    list_of_inputs = []

    # Input 1
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
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

    # Input 2
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75], dtype=np.float32)
    max_output_size = np.array(1, dtype=np.int32)
    iou_threshold = 0.5
    name = "non_max_suppression_1"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = np.array(4, dtype=np.int32)
    iou_threshold = 0.1
    name = "non_max_suppression_2"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = np.array(1, dtype=np.int32)
    iou_threshold = 0.9
    name = "non_max_suppression_3"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    max_output_size = np.array(1, dtype=np.int32)
    iou_threshold = 0.5
    name = "non_max_suppression_4"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different box coordinates
    boxes = np.array([[0.1, 0.2, 0.8, 0.9], [0.2, 0.3, 0.9, 1.0], [0.0, 0.1, 0.7, 0.8]], dtype=np.float32)
    scores = np.array([0.8, 0.9, 0.7], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = 0.6
    name = None

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  Small max_output_size
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

    # Input 8: Boxes with same scores
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.9], dtype=np.float32)
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

    # Input 9: no overlap
    boxes = np.array([[0, 0, 0.5, 0.5], [0.6, 0.6, 1, 1]], dtype=np.float32)
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
    
    # Input 10: All boxes overlap
    boxes = np.array([[0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8]], dtype=np.float32)
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
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.NonMaxSuppression"] = tf_raw_ops_non_max_suppression_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NonMaxSuppression' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NonMaxSuppression'.")

check_valid('tf.raw_ops.NonMaxSuppression', generated_inputs['tf.raw_ops.NonMaxSuppression'], lib="tf", suffix=0)
