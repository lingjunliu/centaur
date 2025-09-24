
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_non_max_suppression_v2_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": "basic_case"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: No overlap
    boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": "no_overlap"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: All boxes overlap
    boxes = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.array(1, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": "all_overlap"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: max_output_size = 0
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75], dtype=np.float32)
    max_output_size = np.array(0, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": "max_output_size_0"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Higher iou_threshold
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.9, dtype=np.float32)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": "higher_iou_threshold"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: Half type
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5]], dtype=np.float16)
    scores = np.array([0.9, 0.75], dtype=np.float16)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float16)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": "half_type"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Normalized coordinates
    boxes = np.array([[0.1, 0.1, 0.2, 0.2], [0.15, 0.15, 0.25, 0.25]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": "normalized_coordinates"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different box order (y2 < y1, x2 < x1)
    boxes = np.array([[1, 1, 0, 0], [1, 1.5, 0, 0.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": "different_box_order"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single box
    boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    max_output_size = np.array(1, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": "single_box"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Several boxes, high threshold to keep all
    boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.array(3, dtype=np.int32)
    iou_threshold = np.array(1.0, dtype=np.float32)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": "high_iou_threshold_keep_all"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Larger number of boxes, max_output_size smaller
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5], [1, 1, 2, 2]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95, 0.8], dtype=np.float32)
    max_output_size = np.array(3, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "name": "larger_boxes_smaller_max_output_size"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.NonMaxSuppressionV2"] = tf_raw_ops_non_max_suppression_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NonMaxSuppressionV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NonMaxSuppressionV2'.")

check_valid('tf.raw_ops.NonMaxSuppressionV2', generated_inputs['tf.raw_ops.NonMaxSuppressionV2'], lib="tf", suffix=0)
