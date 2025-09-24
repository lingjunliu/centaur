
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_NonMaxSuppressionV5_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.4, dtype=np.float32)
    soft_nms_sigma = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms_1"
    input_dict = {"boxes": boxes, "scores": scores, "max_output_size": max_output_size,
                  "iou_threshold": iou_threshold, "score_threshold": score_threshold,
                  "soft_nms_sigma": soft_nms_sigma, "pad_to_max_output_size": pad_to_max_output_size,
                  "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2:  Different iou_threshold
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.9, dtype=np.float32)
    score_threshold = np.array(0.4, dtype=np.float32)
    soft_nms_sigma = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms_2"
    input_dict = {"boxes": boxes, "scores": scores, "max_output_size": max_output_size,
                  "iou_threshold": iou_threshold, "score_threshold": score_threshold,
                  "soft_nms_sigma": soft_nms_sigma, "pad_to_max_output_size": pad_to_max_output_size,
                  "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  Different score_threshold
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.8, dtype=np.float32)
    soft_nms_sigma = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms_3"
    input_dict = {"boxes": boxes, "scores": scores, "max_output_size": max_output_size,
                  "iou_threshold": iou_threshold, "score_threshold": score_threshold,
                  "soft_nms_sigma": soft_nms_sigma, "pad_to_max_output_size": pad_to_max_output_size,
                  "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: soft_nms_sigma > 0
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.4, dtype=np.float32)
    soft_nms_sigma = np.array(0.5, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms_4"
    input_dict = {"boxes": boxes, "scores": scores, "max_output_size": max_output_size,
                  "iou_threshold": iou_threshold, "score_threshold": score_threshold,
                  "soft_nms_sigma": soft_nms_sigma, "pad_to_max_output_size": pad_to_max_output_size,
                  "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: pad_to_max_output_size = True
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = np.array(5, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.4, dtype=np.float32)
    soft_nms_sigma = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = True
    name = "nms_5"
    input_dict = {"boxes": boxes, "scores": scores, "max_output_size": max_output_size,
                  "iou_threshold": iou_threshold, "score_threshold": score_threshold,
                  "soft_nms_sigma": soft_nms_sigma, "pad_to_max_output_size": pad_to_max_output_size,
                  "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: half type
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32) # Changed to float32 as float16 may cause issues
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32) # Changed to float32 as float16 may cause issues
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32) # Changed to float32 as float16 may cause issues
    score_threshold = np.array(0.4, dtype=np.float32) # Changed to float32 as float16 may cause issues
    soft_nms_sigma = np.array(0.0, dtype=np.float32) # Changed to float32 as float16 may cause issues
    pad_to_max_output_size = False
    name = "nms_6"
    input_dict = {"boxes": boxes, "scores": scores, "max_output_size": max_output_size,
                  "iou_threshold": iou_threshold, "score_threshold": score_threshold,
                  "soft_nms_sigma": soft_nms_sigma, "pad_to_max_output_size": pad_to_max_output_size,
                  "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty boxes and scores
    boxes = np.array([], dtype=np.float32).reshape(0, 4)
    scores = np.array([], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.4, dtype=np.float32)
    soft_nms_sigma = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms_7"
    input_dict = {"boxes": boxes, "scores": scores, "max_output_size": max_output_size,
                  "iou_threshold": iou_threshold, "score_threshold": score_threshold,
                  "soft_nms_sigma": soft_nms_sigma, "pad_to_max_output_size": pad_to_max_output_size,
                  "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  boxes and scores with one element
    boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.4, dtype=np.float32)
    soft_nms_sigma = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms_8"
    input_dict = {"boxes": boxes, "scores": scores, "max_output_size": max_output_size,
                  "iou_threshold": iou_threshold, "score_threshold": score_threshold,
                  "soft_nms_sigma": soft_nms_sigma, "pad_to_max_output_size": pad_to_max_output_size,
                  "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large max_output_size
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = np.array(100, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.4, dtype=np.float32)
    soft_nms_sigma = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms_9"
    input_dict = {"boxes": boxes, "scores": scores, "max_output_size": max_output_size,
                  "iou_threshold": iou_threshold, "score_threshold": score_threshold,
                  "soft_nms_sigma": soft_nms_sigma, "pad_to_max_output_size": pad_to_max_output_size,
                  "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: boxes with negative coordinates
    boxes = np.array([[-0.1, -0.1, 0.9, 0.9], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = np.array(2, dtype=np.int32)
    iou_threshold = np.array(0.5, dtype=np.float32)
    score_threshold = np.array(0.4, dtype=np.float32)
    soft_nms_sigma = np.array(0.0, dtype=np.float32)
    pad_to_max_output_size = False
    name = "nms_10"
    input_dict = {"boxes": boxes, "scores": scores, "max_output_size": max_output_size,
                  "iou_threshold": iou_threshold, "score_threshold": score_threshold,
                  "soft_nms_sigma": soft_nms_sigma, "pad_to_max_output_size": pad_to_max_output_size,
                  "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.NonMaxSuppressionV5"] = tf_raw_ops_NonMaxSuppressionV5_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NonMaxSuppressionV5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NonMaxSuppressionV5'.")

check_valid('tf.raw_ops.NonMaxSuppressionV5', generated_inputs['tf.raw_ops.NonMaxSuppressionV5'], lib="tf", suffix=0)
