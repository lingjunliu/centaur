
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_non_max_suppression_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case with 2 boxes
    list_of_inputs.append({
        'boxes': np.array([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 1.1, 1.1]], dtype=np.float32),
        'scores': np.array([0.9, 0.8], dtype=np.float32),
        'max_output_size': int(1),
        'iou_threshold': float(0.5),
        'score_threshold': float(0.0),
        'name': 'nms_1'
    })

    # Input 2: Multiple non-overlapping boxes
    list_of_inputs.append({
        'boxes': np.array([[float(i), float(i), float(i + 1), float(i + 1)] for i in range(10)], dtype=np.float32),
        'scores': np.array([float(i) / 10.0 for i in range(10)], dtype=np.float32),
        'max_output_size': int(5),
        'iou_threshold': float(0.5),
        'score_threshold': float(0.1),
        'name': 'nms_2'
    })

    # Input 3: Heavy overlap, low max_output_size
    list_of_inputs.append({
        'boxes': np.array([[0.0, 0.0, 1.0, 1.0]] * 5, dtype=np.float32),
        'scores': np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        'max_output_size': int(2),
        'iou_threshold': float(0.1),
        'score_threshold': float(0.0),
        'name': 'nms_3'
    })

    # Input 4: Negative coordinates and negative scores
    list_of_inputs.append({
        'boxes': np.array([[-10.0, -10.0, 0.0, 0.0], [-9.0, -9.0, 1.0, 1.0]], dtype=np.float32),
        'scores': np.array([-10.0, -5.0], dtype=np.float32),
        'max_output_size': int(2),
        'iou_threshold': float(0.8),
        'score_threshold': float(-15.0),
        'name': 'nms_4'
    })

    # Input 5: Large scale coordinates, high iou_threshold
    list_of_inputs.append({
        'boxes': np.array([[100.0, 100.0, 200.0, 200.0], [120.0, 120.0, 220.0, 220.0], [300.0, 300.0, 400.0, 400.0]], dtype=np.float32),
        'scores': np.array([0.9, 0.75, 0.6], dtype=np.float32),
        'max_output_size': int(3),
        'iou_threshold': float(0.9),
        'score_threshold': float(0.5),
        'name': 'nms_5'
    })

    # Input 6: Zero boxes (empty case)
    list_of_inputs.append({
        'boxes': np.empty((0, 4), dtype=np.float32),
        'scores': np.empty((0,), dtype=np.float32),
        'max_output_size': int(5),
        'iou_threshold': float(0.5),
        'score_threshold': float(0.0),
        'name': 'nms_6'
    })

    # Input 7: Only one box
    list_of_inputs.append({
        'boxes': np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        'scores': np.array([0.5], dtype=np.float32),
        'max_output_size': int(10),
        'iou_threshold': float(0.5),
        'score_threshold': float(0.1),
        'name': 'nms_7'
    })

    # Input 8: High score_threshold (all pruned)
    list_of_inputs.append({
        'boxes': np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]], dtype=np.float32),
        'scores': np.array([0.3, 0.4], dtype=np.float32),
        'max_output_size': int(2),
        'iou_threshold': float(0.5),
        'score_threshold': float(0.5),
        'name': 'nms_8'
    })

    # Input 9: iou_threshold = 0.0 (aggressive suppression)
    list_of_inputs.append({
        'boxes': np.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]], dtype=np.float32),
        'scores': np.array([0.9, 0.8], dtype=np.float32),
        'max_output_size': int(2),
        'iou_threshold': float(0.0),
        'score_threshold': float(0.1),
        'name': 'nms_9'
    })

    # Input 10: iou_threshold = 1.0 (no suppression for overlaps)
    list_of_inputs.append({
        'boxes': np.array([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]], dtype=np.float32),
        'scores': np.array([0.9, 0.8], dtype=np.float32),
        'max_output_size': int(2),
        'iou_threshold': float(1.0),
        'score_threshold': float(0.1),
        'name': 'nms_10'
    })

    return list_of_inputs

generated_inputs["tf.image.non_max_suppression"] = tf_image_non_max_suppression_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.non_max_suppression' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.non_max_suppression'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.non_max_suppression', generated_inputs['tf.image.non_max_suppression'], lib="tf", suffix=0)
