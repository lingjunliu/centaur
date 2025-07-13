
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_NonMaxSuppressionV3_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1], [0.0, -0.1, 1.0, 0.9]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.5], dtype=np.float32)
    max_output_size = np.int32(2)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.0)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2:  Different boxes and scores
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.5], [0.5, 0.0, 1.5, 1.0]], dtype=np.float32)
    scores = np.array([0.8, 0.9, 0.7], dtype=np.float32)
    max_output_size = np.int32(3)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.0)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Higher IOU threshold
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = np.int32(2)
    iou_threshold = np.float32(0.9)
    score_threshold = np.float32(0.0)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Score threshold active
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1], [0.0, 0.2, 1.0, 1.2]], dtype=np.float32)
    scores = np.array([0.9, 0.7, 0.3], dtype=np.float32)
    max_output_size = np.int32(3)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.5)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: max_output_size is 1
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = np.int32(1)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.0)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Overlapping boxes
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.9, 0.9]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = np.int32(2)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.0)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: No overlapping boxes
    boxes = np.array([[0.0, 0.0, 0.1, 0.1], [0.9, 0.9, 1.0, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = np.int32(2)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.0)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_NonMaxSuppressionV3_inputs()
generated_inputs["tf.raw_ops.NonMaxSuppressionV3"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.NonMaxSuppressionV3"].append({
        "kwargs": input_dict
    })

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NonMaxSuppressionV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NonMaxSuppressionV3'.")

check_valid('tf.raw_ops.NonMaxSuppressionV3', generated_inputs['tf.raw_ops.NonMaxSuppressionV3'], lib="tf", suffix=0)
