
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_non_max_suppression_with_scores_inputs():
    list_of_inputs = []

    # Input 1
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = 2
    iou_threshold = 0.5
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms1"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "soft_nms_sigma": soft_nms_sigma,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = 0.5
    soft_nms_sigma = 0.0
    name = "nms2"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "soft_nms_sigma": soft_nms_sigma,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = 1
    iou_threshold = 0.5
    score_threshold = 0.8
    soft_nms_sigma = 0.0
    name = "nms3"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "soft_nms_sigma": soft_nms_sigma,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Soft NMS
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5 # Ignored
    score_threshold = 0.0
    soft_nms_sigma = 0.5
    name = "nms4"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "soft_nms_sigma": soft_nms_sigma,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5: Different box coordinates
    boxes = np.array([[0.1, 0.2, 0.6, 0.8], [0.3, 0.4, 0.7, 0.9], [0.5, 0.6, 0.9, 1.0]], dtype=np.float32)
    scores = np.array([0.8, 0.7, 0.6], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.4
    score_threshold = 0.5
    soft_nms_sigma = 0.0
    name = "nms5"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "soft_nms_sigma": soft_nms_sigma,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative score threshold
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = -1.0
    soft_nms_sigma = 0.0
    name = "nms6"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "soft_nms_sigma": soft_nms_sigma,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: IOU Threshold of 1.0
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 1.0
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms7"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "soft_nms_sigma": soft_nms_sigma,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero boxes
    boxes = np.array([], dtype=np.float32).reshape(0, 4)
    scores = np.array([], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms8"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "soft_nms_sigma": soft_nms_sigma,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Absolute coordinates
    boxes = np.array([[10, 20, 30, 40], [15, 25, 35, 45], [25, 35, 45, 55]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms9"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "soft_nms_sigma": soft_nms_sigma,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large iou
    boxes = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.99
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms10"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "soft_nms_sigma": soft_nms_sigma,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.non_max_suppression_with_scores"] = tf_image_non_max_suppression_with_scores_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.non_max_suppression_with_scores' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.non_max_suppression_with_scores'.")

check_valid('tf.image.non_max_suppression_with_scores', generated_inputs['tf.image.non_max_suppression_with_scores'], lib="tf", suffix=0)
