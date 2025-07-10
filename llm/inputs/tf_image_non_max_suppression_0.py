
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_non_max_suppression_inputs():
    list_of_inputs = []

    # Input 1
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = np.int32(2)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.0)
    name = "nms1"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75], dtype=np.float32)
    max_output_size = np.int32(1)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.5)
    name = "nms2"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = np.int32(4)
    iou_threshold = np.float32(0.1)
    score_threshold = np.float32(0.0)
    name = "nms3"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75], dtype=np.float32)
    max_output_size = np.int32(2)
    iou_threshold = np.float32(0.9)
    score_threshold = np.float32(0.0)
    name = "nms4"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    max_output_size = np.int32(1)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.0)
    name = "nms5"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    boxes = np.array([[0, 0, 0.5, 0.5], [0.1, 0.1, 0.6, 0.6], [0.2, 0.2, 0.7, 0.7]], dtype=np.float32)
    scores = np.array([0.8, 0.9, 0.7], dtype=np.float32)
    max_output_size = np.int32(2)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(-1.0)
    name = "nms6"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = np.int32(1)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.9)
    name = "nms7"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = np.int32(10)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(-np.inf)
    name = "nms8"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty boxes
    boxes = np.array([], dtype=np.float32).reshape(0, 4)
    scores = np.array([], dtype=np.float32)
    max_output_size = np.int32(10)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(-np.inf)
    name = "nms9"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different box coordinates
    boxes = np.array([[0.1, 0.2, 0.6, 0.8], [0.2, 0.3, 0.7, 0.9], [0.3, 0.4, 0.8, 1.0]], dtype=np.float32)
    scores = np.array([0.7, 0.8, 0.9], dtype=np.float32)
    max_output_size = np.int32(3)
    iou_threshold = np.float32(0.3)
    score_threshold = np.float32(0.5)
    name = "nms10"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.non_max_suppression"] = tf_image_non_max_suppression_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.non_max_suppression' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.non_max_suppression'.")

check_valid('tf.image.non_max_suppression', generated_inputs['tf.image.non_max_suppression'], lib="tf", suffix=0)
