
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_combined_non_max_suppression_inputs():
    list_of_inputs = []

    # Input 1
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.2, 0.2, 0.4, 0.4]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.array(1, dtype=np.int32)
    max_total_size = np.array(2, dtype=np.int32)
    iou_threshold = 0.5
    score_threshold = 0.0
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
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.2, 0.2, 0.4, 0.4]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.array(2, dtype=np.int32)
    max_total_size = np.array(1, dtype=np.int32)
    iou_threshold = 0.5
    score_threshold = 0.0
    pad_per_class = True
    clip_boxes = False
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
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.2, 0.2, 0.4, 0.4]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.array(1, dtype=np.int32)
    max_total_size = np.array(2, dtype=np.int32)
    iou_threshold = 0.3
    score_threshold = 0.1
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
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.2, 0.2, 0.4, 0.4]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.array(1, dtype=np.int32)
    max_total_size = np.array(2, dtype=np.int32)
    iou_threshold = 0.5
    score_threshold = -0.5
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
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.2, 0.2, 0.4, 0.4]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.array(1, dtype=np.int32)
    max_total_size = np.array(2, dtype=np.int32)
    iou_threshold = 0.9
    score_threshold = 0.9
    pad_per_class = True
    clip_boxes = False
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
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.2, 0.2, 0.4, 0.4]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.array(1, dtype=np.int32)
    max_total_size = np.array(2, dtype=np.int32)
    iou_threshold = 0.01
    score_threshold = -1.0
    pad_per_class = False
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
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.2, 0.2, 0.4, 0.4]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.array(2, dtype=np.int32)
    max_total_size = np.array(4, dtype=np.int32)
    iou_threshold = 0.7
    score_threshold = 0.3
    pad_per_class = True
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
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.2, 0.2, 0.4, 0.4]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.array(2, dtype=np.int32)
    max_total_size = np.array(4, dtype=np.int32)
    iou_threshold = 0.2
    score_threshold = 0.4
    pad_per_class = True
    clip_boxes = False
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
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.2, 0.2, 0.4, 0.4]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.array(1, dtype=np.int32)
    max_total_size = np.array(2, dtype=np.int32)
    iou_threshold = 0.5
    score_threshold = 0.0
    pad_per_class = False
    clip_boxes = False
    name = None

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
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.2, 0.2, 0.4, 0.4]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.array(1, dtype=np.int32)
    max_total_size = np.array(2, dtype=np.int32)
    iou_threshold = 0.5
    score_threshold = 0.0
    pad_per_class = False
    clip_boxes = True
    name = ""

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
generated_inputs["tf.image.combined_non_max_suppression"] = tf_image_combined_non_max_suppression_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.combined_non_max_suppression' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.combined_non_max_suppression'.")

check_valid('tf.image.combined_non_max_suppression', generated_inputs['tf.image.combined_non_max_suppression'], lib="tf", suffix=0)
