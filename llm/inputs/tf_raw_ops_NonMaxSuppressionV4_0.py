
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_non_max_suppression_v4_inputs():
    list_of_inputs = []

    # Input 1: Basic example with some overlap
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.8], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = None

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_to_max_output_size": pad_to_max_output_size,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: No overlap
    boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.array(3, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = None

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_to_max_output_size": pad_to_max_output_size,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: High overlap, only one box survives
    boxes = np.array([[0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.array(3, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = None

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_to_max_output_size": pad_to_max_output_size,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Score threshold eliminates some boxes
    boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], dtype=np.float32)
    scores = np.array([0.9, 0.4, 0.7], dtype=np.float32)
    max_output_size = np.array(3, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.5, dtype=np.float32)
    pad_to_max_output_size = False
    name = None

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_to_max_output_size": pad_to_max_output_size,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: max_output_size limits the number of selected boxes
    boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = None

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_to_max_output_size": pad_to_max_output_size,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: pad_to_max_output_size is True
    boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = np.array(3, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = True
    name = None

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_to_max_output_size": pad_to_max_output_size,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different box coordinates
    boxes = np.array([[0.1, 0.2, 0.5, 0.8], [0.6, 0.7, 0.9, 0.95]], dtype=np.float32)
    scores = np.array([0.85, 0.92], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.3, dtype=np.float32)
    score_threshold = np.array(0.1, dtype=np.float32)
    pad_to_max_output_size = False
    name = None

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_to_max_output_size": pad_to_max_output_size,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All boxes have same score.
    boxes = np.array([[0, 0, 1, 1], [0.2, 0.2, 0.8, 0.8]], dtype=np.float32)
    scores = np.array([0.9, 0.9], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = None

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_to_max_output_size": pad_to_max_output_size,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Using Half type
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float16)
    scores = np.array([0.9, 0.75, 0.8], dtype=np.float16)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float16)
    score_threshold = np.array(0.0, dtype=np.float16)
    pad_to_max_output_size = False
    name = None

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_to_max_output_size": pad_to_max_output_size,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty boxes.
    boxes = np.array([], dtype=np.float32).reshape(0, 4)
    scores = np.array([], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = None

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_to_max_output_size": pad_to_max_output_size,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.NonMaxSuppressionV4"] = tf_raw_ops_non_max_suppression_v4_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NonMaxSuppressionV4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NonMaxSuppressionV4'.")

check_valid('tf.raw_ops.NonMaxSuppressionV4', generated_inputs['tf.raw_ops.NonMaxSuppressionV4'], lib="tf", suffix=0)
