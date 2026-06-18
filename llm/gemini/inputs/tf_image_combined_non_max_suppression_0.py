
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_combined_non_max_suppression_inputs():
    list_of_inputs = []
    
    np.random.seed(42)
    
    def get_valid_boxes(shape):
        boxes = np.random.rand(*shape).astype(np.float32)
        y1 = np.minimum(boxes[..., 0], boxes[..., 2])
        y2 = np.maximum(boxes[..., 0], boxes[..., 2])
        x1 = np.minimum(boxes[..., 1], boxes[..., 3])
        x2 = np.maximum(boxes[..., 1], boxes[..., 3])
        boxes[..., 0] = y1
        boxes[..., 1] = x1
        boxes[..., 2] = y2
        boxes[..., 3] = x2
        return boxes

    # Input 1
    boxes = get_valid_boxes([2, 5, 1, 4])
    scores = np.random.rand(2, 5, 3).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 2,
        'max_total_size': 5,
        'iou_threshold': 0.5,
        'score_threshold': 0.1,
        'pad_per_class': False,
        'clip_boxes': True,
        'name': 'nms_1'
    })

    # Input 2
    boxes = get_valid_boxes([1, 10, 2, 4])
    scores = np.random.rand(1, 10, 2).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 3,
        'max_total_size': 4,
        'iou_threshold': 0.6,
        'score_threshold': 0.0,
        'pad_per_class': True,
        'clip_boxes': False,
        'name': 'nms_2'
    })

    # Input 3
    boxes = get_valid_boxes([3, 4, 1, 4])
    scores = np.random.rand(3, 4, 1).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 1,
        'max_total_size': 2,
        'iou_threshold': 0.3,
        'score_threshold': 0.5,
        'pad_per_class': False,
        'clip_boxes': True,
        'name': 'nms_3'
    })

    # Input 4
    boxes = get_valid_boxes([2, 8, 4, 4])
    scores = np.random.rand(2, 8, 4).astype(np.float32) - 0.5
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 4,
        'max_total_size': 10,
        'iou_threshold': 0.7,
        'score_threshold': -0.2,
        'pad_per_class': True,
        'clip_boxes': True,
        'name': 'nms_4'
    })

    # Input 5
    boxes = get_valid_boxes([1, 1, 1, 4])
    scores = np.random.rand(1, 1, 1).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 1,
        'max_total_size': 1,
        'iou_threshold': 0.5,
        'score_threshold': 0.0,
        'pad_per_class': False,
        'clip_boxes': True,
        'name': 'nms_5'
    })

    # Input 6
    boxes = get_valid_boxes([4, 15, 1, 4])
    scores = np.random.rand(4, 15, 2).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 5,
        'max_total_size': 10,
        'iou_threshold': 0.4,
        'score_threshold': 0.2,
        'pad_per_class': False,
        'clip_boxes': False,
        'name': 'nms_6'
    })

    # Input 7
    boxes = get_valid_boxes([2, 2, 2, 4])
    scores = np.random.rand(2, 2, 2).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 2,
        'max_total_size': 4,
        'iou_threshold': 0.1,
        'score_threshold': 0.9,
        'pad_per_class': True,
        'clip_boxes': True,
        'name': 'nms_7'
    })

    # Input 8
    boxes = get_valid_boxes([1, 5, 1, 4])
    scores = (np.random.rand(1, 5, 3).astype(np.float32) - 1.5)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 2,
        'max_total_size': 6,
        'iou_threshold': 0.9,
        'score_threshold': -1.0,
        'pad_per_class': False,
        'clip_boxes': False,
        'name': 'nms_8'
    })

    # Input 9
    boxes = get_valid_boxes([5, 12, 1, 4])
    scores = np.random.rand(5, 12, 1).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 3,
        'max_total_size': 15,
        'iou_threshold': 0.5,
        'score_threshold': 0.05,
        'pad_per_class': True,
        'clip_boxes': True,
        'name': 'nms_9'
    })

    # Input 10
    boxes = get_valid_boxes([2, 6, 3, 4])
    scores = np.random.rand(2, 6, 3).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 1,
        'max_total_size': 3,
        'iou_threshold': 0.8,
        'score_threshold': 0.3,
        'pad_per_class': False,
        'clip_boxes': False,
        'name': 'nms_10'
    })

    return list_of_inputs

generated_inputs["tf.image.combined_non_max_suppression"] = tf_image_combined_non_max_suppression_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.combined_non_max_suppression' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.combined_non_max_suppression'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.combined_non_max_suppression', generated_inputs['tf.image.combined_non_max_suppression'], lib="tf", suffix=0)
