
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_non_max_suppression_v3_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = np.array(3, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.4, dtype=np.float32)

    input_dict = {
        "name": "basic_case",
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: No boxes
    boxes = np.array([], dtype=np.float32).reshape(0, 4)
    scores = np.array([], dtype=np.float32)
    max_output_size = np.array(5, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.4, dtype=np.float32)

    input_dict = {
        "name": "no_boxes",
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: All boxes overlap
    boxes = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(1.0, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)

    input_dict = {
        "name": "all_overlap",
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Score threshold eliminates all boxes
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5]], dtype=np.float32)
    scores = np.array([0.1, 0.2], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.3, dtype=np.float32)

    input_dict = {
        "name": "high_score_threshold",
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Max output size = 0
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.7], dtype=np.float32)
    max_output_size = np.array(0, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.3, dtype=np.float32)

    input_dict = {
        "name": "zero_max_output",
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different box coordinates
    boxes = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 0.2, 0.3]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.array(3, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)

    input_dict = {
        "name": "different_boxes",
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: IOU Threshold = 0
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.7], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.0, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)

    input_dict = {
        "name": "zero_iou_threshold",
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Only 1 box
    boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    max_output_size = np.array(1, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)

    input_dict = {
        "name": "single_box",
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Half type
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5]], dtype=np.float16)
    scores = np.array([0.9, 0.7], dtype=np.float16)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float16)
    score_threshold = np.array(0.0, dtype=np.float16)

    input_dict = {
        "name": "half_type",
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Identical boxes different scores
    boxes = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.array(3, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)

    input_dict = {
        "name": "identical_boxes_different_scores",
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.NonMaxSuppressionV3"] = tf_raw_ops_non_max_suppression_v3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NonMaxSuppressionV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NonMaxSuppressionV3'.")

check_valid('tf.raw_ops.NonMaxSuppressionV3', generated_inputs['tf.raw_ops.NonMaxSuppressionV3'], lib="tf", suffix=0)
