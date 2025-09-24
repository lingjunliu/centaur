
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_NonMaxSuppressionV4_inputs():
    list_of_inputs = []

    # Input 1
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms1"

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

    # Input 2
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = np.array(4, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = True
    name = "nms2"

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

    # Input 3
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = np.array(1, dtype=np.int32)
    iou_threshold = np.array(0.1, dtype=np.float32)
    score_threshold = np.array(0.5, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms3"

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

   # Input 4
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = np.array(4, dtype=np.int32)
    iou_threshold = np.array(0.9, dtype=np.float32)
    score_threshold = np.array(0.5, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms4"

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

    # Input 5
    boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    max_output_size = np.array(1, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms5"

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

   # Input 6
    boxes = np.array([[0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8], [0.3, 0.3, 0.7, 0.7]], dtype=np.float32)
    scores = np.array([0.8, 0.7, 0.6], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.4, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms6"

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

    # Input 7
    boxes = np.array([[0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8], [0.3, 0.3, 0.7, 0.7]], dtype=np.float32)
    scores = np.array([0.8, 0.7, 0.6], dtype=np.float32)
    max_output_size = np.array(5, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.4, dtype=np.float32)
    pad_to_max_output_size = True
    name = "nms7"

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

    # Input 8, half type
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float16)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float16)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float16)
    score_threshold = np.array(0.0, dtype=np.float16)
    pad_to_max_output_size = False
    name = "nms8"

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
    
    # Input 9, all boxes are the same
    boxes = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms9"

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

    # Input 10, zero boxes
    boxes = np.array([], dtype=np.float32).reshape(0, 4)
    scores = np.array([], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms10"

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
    
    # Input 11: High IOU threshold
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.99, dtype=np.float32)  # Very high threshold
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms11"

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
    
    # Input 12: Max output size is zero
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = np.array(0, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms12"

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
generated_inputs["tf.raw_ops.NonMaxSuppressionV4"] = tf_raw_ops_NonMaxSuppressionV4_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NonMaxSuppressionV4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NonMaxSuppressionV4'.")

check_valid('tf.raw_ops.NonMaxSuppressionV4', generated_inputs['tf.raw_ops.NonMaxSuppressionV4'], lib="tf", suffix=0)
