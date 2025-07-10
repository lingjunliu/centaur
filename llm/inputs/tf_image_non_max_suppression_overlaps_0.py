
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_non_max_suppression_overlaps_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    overlaps = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.1], [0.2, 0.1, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.int32(2)
    overlap_threshold = np.float32(0.5)
    score_threshold = np.float32(0.6)
    name = "nms_overlaps_1"
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: No overlap
    overlaps = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.int32(3)
    overlap_threshold = np.float32(0.5)
    score_threshold = np.float32(0.6)
    name = "nms_overlaps_2"
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: High overlap, only one box remains
    overlaps = np.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.7], [0.8, 0.7, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.int32(1)
    overlap_threshold = np.float32(0.5)
    score_threshold = np.float32(0.0)
    name = "nms_overlaps_3"
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: All boxes below score threshold
    overlaps = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.1], [0.2, 0.1, 1.0]], dtype=np.float32)
    scores = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    max_output_size = np.int32(3)
    overlap_threshold = np.float32(0.5)
    score_threshold = np.float32(0.4)
    name = "nms_overlaps_4"
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: max_output_size = 0
    overlaps = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.1], [0.2, 0.1, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.int32(0)
    overlap_threshold = np.float32(0.5)
    score_threshold = np.float32(0.6)
    name = "nms_overlaps_5"
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large number of boxes
    num_boxes = 10
    overlaps = np.eye(num_boxes, dtype=np.float32)
    scores = np.linspace(1.0, 0.1, num_boxes, dtype=np.float32)
    max_output_size = np.int32(5)
    overlap_threshold = np.float32(0.5)
    score_threshold = np.float32(0.0)
    name = "nms_overlaps_6"
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All scores are the same
    overlaps = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.1], [0.2, 0.1, 1.0]], dtype=np.float32)
    scores = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    max_output_size = np.int32(2)
    overlap_threshold = np.float32(0.5)
    score_threshold = np.float32(0.6)
    name = "nms_overlaps_7"
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative score_threshold
    overlaps = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.1], [0.2, 0.1, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.int32(2)
    overlap_threshold = np.float32(0.5)
    score_threshold = np.float32(-1.0)
    name = "nms_overlaps_8"
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: overlap_threshold = 1.0
    overlaps = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.1], [0.2, 0.1, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.int32(3)
    overlap_threshold = np.float32(1.0)
    score_threshold = np.float32(0.6)
    name = "nms_overlaps_9"
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: overlap_threshold = 0.0
    overlaps = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.1], [0.2, 0.1, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.int32(3)
    overlap_threshold = np.float32(0.0)
    score_threshold = np.float32(0.6)
    name = "nms_overlaps_10"
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.non_max_suppression_overlaps"] = tf_image_non_max_suppression_overlaps_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.non_max_suppression_overlaps' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.non_max_suppression_overlaps'.")

check_valid('tf.image.non_max_suppression_overlaps', generated_inputs['tf.image.non_max_suppression_overlaps'], lib="tf", suffix=0)
