
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_combined_non_max_suppression_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.CombinedNonMaxSuppression function.
    """
    list_of_inputs = []

    # Input 1: Basic case with shared boxes
    input_dict_1 = {
        'boxes': np.array([
            [[[0.1, 0.1, 0.3, 0.3]], [[0.12, 0.12, 0.32, 0.32]],
             [[0.5, 0.5, 0.7, 0.7]], [[0.9, 0.9, 1.0, 1.0]],
             [[0.52, 0.52, 0.72, 0.72]]]
        ], dtype=np.float32),
        'scores': np.array([
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.05, 0.95], [0.6, 0.4]]
        ], dtype=np.float32),
        'max_output_size_per_class': np.array(2, dtype=np.int32),
        'max_total_size': np.array(3, dtype=np.int32),
        'iou_threshold': np.array(0.5, dtype=np.float32),
        'score_threshold': np.array(0.1, dtype=np.float32),
        'pad_per_class': False,
        'clip_boxes': True,
        'name': 'basic_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Class-specific boxes and absolute coordinates
    input_dict_2 = {
        'boxes': np.array([
            [[[10, 10, 30, 30], [0, 50, 20, 80], [40, 40, 50, 50]],
             [[11, 11, 31, 31], [5, 55, 25, 85], [41, 41, 51, 51]],
             [[50, 50, 70, 70], [100, 100, 120, 120], [80, 80, 90, 90]],
             [[90, 90, 100, 100], [150, 150, 170, 170], [120, 120, 130, 130]]]
        ], dtype=np.float32),
        'scores': np.array([
            [[0.9, 0.8, 0.7], [0.85, 0.75, 0.65], [0.8, 0.9, 0.6], [0.75, 0.85, 0.55]]
        ], dtype=np.float32),
        'max_output_size_per_class': np.array(2, dtype=np.int32),
        'max_total_size': np.array(10, dtype=np.int32),
        'iou_threshold': np.array(0.5, dtype=np.float32),
        'score_threshold': np.array(0.6, dtype=np.float32),
        'pad_per_class': True,
        'clip_boxes': False,
        'name': 'class_specific'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Larger batch size
    input_dict_3 = {
        'boxes': np.random.rand(2, 10, 1, 4).astype(np.float32),
        'scores': np.random.rand(2, 10, 3).astype(np.float32),
        'max_output_size_per_class': np.array(3, dtype=np.int32),
        'max_total_size': np.array(5, dtype=np.int32),
        'iou_threshold': np.array(0.4, dtype=np.float32),
        'score_threshold': np.array(0.2, dtype=np.float32),
        'pad_per_class': False,
        'clip_boxes': True,
        'name': 'batch_size_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: High max_total_size
    input_dict_4 = {
        'boxes': np.random.rand(1, 20, 1, 4).astype(np.float32),
        'scores': np.random.rand(1, 20, 5).astype(np.float32),
        'max_output_size_per_class': np.array(5, dtype=np.int32),
        'max_total_size': np.array(20, dtype=np.int32),
        'iou_threshold': np.array(0.5, dtype=np.float32),
        'score_threshold': np.array(0.1, dtype=np.float32),
        'pad_per_class': False,
        'clip_boxes': True,
        'name': 'high_max_total_size'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Low score_threshold & low iou_threshold
    input_dict_5 = {
        'boxes': np.random.rand(1, 15, 1, 4).astype(np.float32),
        'scores': np.random.rand(1, 15, 2).astype(np.float32),
        'max_output_size_per_class': np.array(5, dtype=np.int32),
        'max_total_size': np.array(10, dtype=np.int32),
        'iou_threshold': np.array(0.2, dtype=np.float32),
        'score_threshold': np.array(0.01, dtype=np.float32),
        'pad_per_class': False,
        'clip_boxes': True,
        'name': 'low_thresholds'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: High score_threshold
    input_dict_6 = {
        'boxes': np.random.rand(1, 10, 1, 4).astype(np.float32),
        'scores': np.random.rand(1, 10, 3).astype(np.float32),
        'max_output_size_per_class': np.array(3, dtype=np.int32),
        'max_total_size': np.array(5, dtype=np.int32),
        'iou_threshold': np.array(0.5, dtype=np.float32),
        'score_threshold': np.array(0.95, dtype=np.float32),
        'pad_per_class': False,
        'clip_boxes': True,
        'name': 'high_score_threshold'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: High iou_threshold
    input_dict_7 = {
        'boxes': np.array([
            [[[0.1, 0.1, 0.3, 0.3]], [[0.11, 0.11, 0.31, 0.31]],
             [[0.12, 0.12, 0.32, 0.32]], [[0.5, 0.5, 0.6, 0.6]]]
        ], dtype=np.float32),
        'scores': np.array([[[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]]], dtype=np.float32),
        'max_output_size_per_class': np.array(3, dtype=np.int32),
        'max_total_size': np.array(4, dtype=np.int32),
        'iou_threshold': np.array(0.9, dtype=np.float32),
        'score_threshold': np.array(0.1, dtype=np.float32),
        'pad_per_class': False,
        'clip_boxes': True,
        'name': 'high_iou_threshold'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: pad_per_class=True with clipping by max_total_size
    input_dict_8 = {
        'boxes': np.random.rand(1, 15, 1, 4).astype(np.float32),
        'scores': np.random.rand(1, 15, 3).astype(np.float32),
        'max_output_size_per_class': np.array(4, dtype=np.int32),
        'max_total_size': np.array(5, dtype=np.int32),
        'iou_threshold': np.array(0.5, dtype=np.float32),
        'score_threshold': np.array(0.1, dtype=np.float32),
        'pad_per_class': True,
        'clip_boxes': True,
        'name': 'pad_and_clip'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Coords outside [0,1] with clip_boxes=True
    input_dict_9 = {
        'boxes': np.array([[[[-0.1, -0.2, 0.5, 0.6]], [[0.8, 0.9, 1.1, 1.2]], [[-0.2, 0.2, 1.2, 0.8]]]], dtype=np.float32),
        'scores': np.random.rand(1, 3, 2).astype(np.float32),
        'max_output_size_per_class': np.array(2, dtype=np.int32),
        'max_total_size': np.array(3, dtype=np.int32),
        'iou_threshold': np.array(0.5, dtype=np.float32),
        'score_threshold': np.array(0.1, dtype=np.float32),
        'clip_boxes': True,
        'pad_per_class': False,
        'name': 'clip_out_of_bounds'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Coords outside [0,1] with clip_boxes=False
    input_dict_10 = {
        'boxes': np.array([[[[-0.1, -0.2, 0.5, 0.6]], [[0.8, 0.9, 1.1, 1.2]], [[-0.2, 0.2, 1.2, 0.8]]]], dtype=np.float32),
        'scores': np.random.rand(1, 3, 2).astype(np.float32),
        'max_output_size_per_class': np.array(2, dtype=np.int32),
        'max_total_size': np.array(3, dtype=np.int32),
        'iou_threshold': np.array(0.5, dtype=np.float32),
        'score_threshold': np.array(0.1, dtype=np.float32),
        'clip_boxes': False,
        'pad_per_class': False,
        'name': 'no_clip_out_of_bounds'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: Edge case - No boxes
    input_dict_11 = {
        'boxes': np.zeros((1, 0, 1, 4), dtype=np.float32),
        'scores': np.zeros((1, 0, 3), dtype=np.float32),
        'max_output_size_per_class': np.array(5, dtype=np.int32),
        'max_total_size': np.array(5, dtype=np.int32),
        'iou_threshold': np.array(0.5, dtype=np.float32),
        'score_threshold': np.array(0.5, dtype=np.float32),
        'pad_per_class': False,
        'clip_boxes': True,
        'name': 'no_boxes'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: All scores below threshold
    input_dict_12 = {
        'boxes': np.random.rand(1, 10, 1, 4).astype(np.float32),
        'scores': np.random.uniform(0.0, 0.4, size=(1, 10, 4)).astype(np.float32),
        'max_output_size_per_class': np.array(5, dtype=np.int32),
        'max_total_size': np.array(10, dtype=np.int32),
        'iou_threshold': np.array(0.5, dtype=np.float32),
        'score_threshold': np.array(0.5, dtype=np.float32),
        'pad_per_class': False,
        'clip_boxes': True,
        'name': 'all_scores_below_thresh'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.raw_ops.CombinedNonMaxSuppression"] = get_combined_non_max_suppression_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.CombinedNonMaxSuppression' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.CombinedNonMaxSuppression'.")

check_valid('tf.raw_ops.CombinedNonMaxSuppression', generated_inputs['tf.raw_ops.CombinedNonMaxSuppression'], lib="tf", suffix=0)
