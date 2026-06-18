
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_image_non_max_suppression_with_scores_inputs():
    list_of_inputs = []

    # Case 1: Standard NMS, basic boxes
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = 2
    iou_threshold = 0.5
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms_standard"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 2: Soft NMS enabled
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1], [0.0, 0.2, 1.0, 1.2]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = 0.1
    soft_nms_sigma = 0.5
    name = "nms_soft"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 3: Empty boxes (0 boxes)
    boxes = np.empty((0, 4), dtype=np.float32)
    scores = np.empty((0,), dtype=np.float32)
    max_output_size = 5
    iou_threshold = 0.5
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms_empty"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 4: Negative scores
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]], dtype=np.float32)
    scores = np.array([-0.5, -0.1], dtype=np.float32)
    max_output_size = 2
    iou_threshold = 0.5
    score_threshold = -1.0
    soft_nms_sigma = 0.0
    name = "nms_neg_scores"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 5: Standard NMS, very high IoU threshold
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.01, 1.0, 1.01]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = 2
    iou_threshold = 0.99
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms_high_iou"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 6: Standard NMS, very low IoU threshold
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = 2
    iou_threshold = 0.01
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms_low_iou"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 7: High max_output_size (larger than num_boxes)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]], dtype=np.float32)
    scores = np.array([0.5, 0.6], dtype=np.float32)
    max_output_size = 100
    iou_threshold = 0.5
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms_large_max_out"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 8: Small max_output_size (1 box allowed)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0], [4.0, 4.0, 5.0, 5.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = 1
    iou_threshold = 0.5
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms_small_max_out"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 9: All boxes identical
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms_identical"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 10: Soft NMS with high sigma
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = 2
    iou_threshold = 0.5
    score_threshold = 0.1
    soft_nms_sigma = 1.5
    name = "nms_soft_high_sigma"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 11: Score threshold filtering (only high scores survive)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0], [4.0, 4.0, 5.0, 5.0]], dtype=np.float32)
    scores = np.array([0.9, 0.3, 0.1], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = 0.5
    soft_nms_sigma = 0.0
    name = "nms_score_filter"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    return list_of_inputs

generated_inputs["tf.image.non_max_suppression_with_scores"] = tf_image_non_max_suppression_with_scores_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.non_max_suppression_with_scores' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.non_max_suppression_with_scores'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.non_max_suppression_with_scores', generated_inputs['tf.image.non_max_suppression_with_scores'], lib="tf", suffix=0)
