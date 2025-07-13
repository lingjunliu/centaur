
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_combined_non_max_suppression_inputs():
    list_of_inputs = []

    # Input 1
    boxes = np.array([[[[0.0, 0.0, 1.0, 1.0]]]], dtype=np.float32)
    scores = np.array([[[0.9]]], dtype=np.float32)
    max_output_size_per_class = np.array(10, dtype=np.int32)
    max_total_size = np.array(10, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_per_class = False
    clip_boxes = True
    name = "nms1"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size_per_class": max_output_size_per_class,
        "max_total_size": max_total_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_per_class": pad_per_class,
        "clip_boxes": clip_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    boxes = np.array([[[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1]]]], dtype=np.float32)
    scores = np.array([[[0.9, 0.8]]], dtype=np.float32)
    max_output_size_per_class = np.array(10, dtype=np.int32)
    max_total_size = np.array(10, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_per_class = False
    clip_boxes = True
    name = "nms2"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size_per_class": max_output_size_per_class,
        "max_total_size": max_total_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_per_class": pad_per_class,
        "clip_boxes": clip_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    boxes = np.array([[[[0.0, 0.0, 1.0, 1.0]], [[0.0, 0.1, 1.0, 1.1]]]], dtype=np.float32)
    scores = np.array([[[0.9]], [[0.8]]], dtype=np.float32)
    max_output_size_per_class = np.array(1, dtype=np.int32)
    max_total_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_per_class = False
    clip_boxes = True
    name = "nms3"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size_per_class": max_output_size_per_class,
        "max_total_size": max_total_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_per_class": pad_per_class,
        "clip_boxes": clip_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    boxes = np.array([[[[0.0, 0.0, 1.0, 1.0]]]], dtype=np.float32)
    scores = np.array([[[0.9]]], dtype=np.float32)
    max_output_size_per_class = np.array(1, dtype=np.int32)
    max_total_size = np.array(1, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.5, dtype=np.float32)
    pad_per_class = False
    clip_boxes = True
    name = "nms4"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size_per_class": max_output_size_per_class,
        "max_total_size": max_total_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_per_class": pad_per_class,
        "clip_boxes": clip_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    boxes = np.array([[[[0.0, 0.0, 1.0, 1.0]]]], dtype=np.float32)
    scores = np.array([[[0.9]]], dtype=np.float32)
    max_output_size_per_class = np.array(1, dtype=np.int32)
    max_total_size = np.array(1, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(1.0, dtype=np.float32)
    pad_per_class = False
    clip_boxes = True
    name = "nms5"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size_per_class": max_output_size_per_class,
        "max_total_size": max_total_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_per_class": pad_per_class,
        "clip_boxes": clip_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    boxes = np.array([[[[0.0, 0.0, 1.0, 1.0]]]], dtype=np.float32)
    scores = np.array([[[0.9]]], dtype=np.float32)
    max_output_size_per_class = np.array(1, dtype=np.int32)
    max_total_size = np.array(1, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_per_class = True
    clip_boxes = True
    name = "nms6"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size_per_class": max_output_size_per_class,
        "max_total_size": max_total_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_per_class": pad_per_class,
        "clip_boxes": clip_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    boxes = np.array([[[[0.0, 0.0, 1.0, 1.0]]]], dtype=np.float32)
    scores = np.array([[[0.9]]], dtype=np.float32)
    max_output_size_per_class = np.array(1, dtype=np.int32)
    max_total_size = np.array(1, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_per_class = False
    clip_boxes = False
    name = "nms7"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size_per_class": max_output_size_per_class,
        "max_total_size": max_total_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_per_class": pad_per_class,
        "clip_boxes": clip_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    boxes = np.array([[[[0.0, 0.0, 1.0, 1.0]]]], dtype=np.float32)
    scores = np.array([[[0.9]]], dtype=np.float32)
    max_output_size_per_class = np.array(1, dtype=np.int32)
    max_total_size = np.array(1, dtype=np.int32)
    iou_threshold = np.array(0.9, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_per_class = False
    clip_boxes = True
    name = "nms8"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size_per_class": max_output_size_per_class,
        "max_total_size": max_total_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_per_class": pad_per_class,
        "clip_boxes": clip_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    boxes = np.array([[[[0.0, 0.0, 1.0, 1.0]] * 2]], dtype=np.float32) # two boxes
    scores = np.array([[[0.9, 0.8]]], dtype=np.float32) # two scores
    max_output_size_per_class = np.array(2, dtype=np.int32)
    max_total_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_per_class = False
    clip_boxes = True
    name = "nms9"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size_per_class": max_output_size_per_class,
        "max_total_size": max_total_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_per_class": pad_per_class,
        "clip_boxes": clip_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    boxes = np.array([[[[0.0, 0.0, 1.0, 1.0]], [[0.0, 0.0, 1.0, 1.0]]]], dtype=np.float32)
    scores = np.array([[[0.9]], [[0.8]]], dtype=np.float32)
    max_output_size_per_class = np.array(1, dtype=np.int32)
    max_total_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_per_class = False
    clip_boxes = True
    name = "nms10"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size_per_class": max_output_size_per_class,
        "max_total_size": max_total_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_per_class": pad_per_class,
        "clip_boxes": clip_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Multiple classes
    boxes = np.array([[[[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.9, 0.9]]]], dtype=np.float32)
    scores = np.array([[[0.9, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.array(10, dtype=np.int32)
    max_total_size = np.array(10, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    pad_per_class = False
    clip_boxes = True
    name = "nms11"

    input_dict = {
        "boxes": boxes,
        "scores": scores,
        "max_output_size_per_class": max_output_size_per_class,
        "max_total_size": max_total_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "pad_per_class": pad_per_class,
        "clip_boxes": clip_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_combined_non_max_suppression_inputs()
generated_inputs["tf.raw_ops.CombinedNonMaxSuppression"] = []
for input_dict in inputs:
    new_input_dict = {}
    new_input_dict['boxes'] = input_dict['boxes']
    new_input_dict['scores'] = input_dict['scores']
    new_input_dict['max_output_size_per_class'] = input_dict['max_output_size_per_class']
    new_input_dict['max_total_size'] = input_dict['max_total_size']
    new_input_dict['iou_threshold'] = input_dict['iou_threshold']
    new_input_dict['score_threshold'] = input_dict['score_threshold']
    new_input_dict['pad_per_class'] = input_dict['pad_per_class']
    new_input_dict['clip_boxes'] = input_dict['clip_boxes']
    if 'name' in input_dict:
        new_input_dict['name'] = input_dict['name']
    generated_inputs["tf.raw_ops.CombinedNonMaxSuppression"].append(new_input_dict)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.CombinedNonMaxSuppression' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.CombinedNonMaxSuppression'.")

check_valid('tf.raw_ops.CombinedNonMaxSuppression', generated_inputs['tf.raw_ops.CombinedNonMaxSuppression'], lib="tf", suffix=0)
