
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_non_max_suppression_with_overlaps_inputs():
    list_of_inputs = []

    # Input 1
    overlaps = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    overlap_threshold = np.array(0.4, dtype=np.float32)
    score_threshold = np.array(0.6, dtype=np.float32)
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    overlaps = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.array(3, dtype=np.int32)
    overlap_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.6, dtype=np.float32)
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    overlaps = np.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.7], [0.8, 0.7, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.array(1, dtype=np.int32)
    overlap_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.6, dtype=np.float32)
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    overlaps = np.array([[1.0, 0.5, 0.2, 0.1], [0.5, 1.0, 0.3, 0.2], [0.2, 0.3, 1.0, 0.3], [0.1, 0.2, 0.3, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    max_output_size = np.array(4, dtype=np.int32)
    overlap_threshold = np.array(0.4, dtype=np.float32)
    score_threshold = np.array(0.5, dtype=np.float32)
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    overlaps = np.eye(5, dtype=np.float32)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    max_output_size = np.array(5, dtype=np.int32)
    overlap_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    overlaps = np.ones((5, 5), dtype=np.float32)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    max_output_size = np.array(1, dtype=np.int32)
    overlap_threshold = np.array(0.0, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    overlaps = np.zeros((3, 3), dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = np.array(3, dtype=np.int32)
    overlap_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(1.0, dtype=np.float32)
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    overlaps = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = np.array(1, dtype=np.int32)
    overlap_threshold = np.array(1.0, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    overlaps = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    overlap_threshold = np.array(0.0, dtype=np.float32)
    score_threshold = np.array(1.0, dtype=np.float32)
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    overlaps = np.array([[1.0]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    max_output_size = np.array(1, dtype=np.int32)
    overlap_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.0, dtype=np.float32)
    input_dict = {"overlaps": overlaps, "scores": scores, "max_output_size": max_output_size, "overlap_threshold": overlap_threshold, "score_threshold": score_threshold, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_non_max_suppression_with_overlaps_inputs()
generated_inputs["tf.raw_ops.NonMaxSuppressionWithOverlaps"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.NonMaxSuppressionWithOverlaps"].append(input_dict)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NonMaxSuppressionWithOverlaps' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NonMaxSuppressionWithOverlaps'.")

check_valid('tf.raw_ops.NonMaxSuppressionWithOverlaps', generated_inputs['tf.raw_ops.NonMaxSuppressionWithOverlaps'], lib="tf", suffix=0)
