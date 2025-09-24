
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_tf_image_combined_non_max_suppression_inputs():
    """
    Generates a list of valid inputs for tf.image.combined_non_max_suppression.
    """
    list_of_inputs = []

    # Input 1: Basic case with shared boxes (q=1)
    input_dict = {
        'boxes': np.random.rand(1, 5, 1, 4).astype(np.float32),
        'scores': np.array([[[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]]).astype(np.float32),
        'max_output_size_per_class': 2,
        'max_total_size': 3,
        'iou_threshold': 0.5,
        'score_threshold': 0.1,
        'pad_per_class': False,
        'clip_boxes': True,
        'name': 'basic_shared_boxes'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Class-specific boxes (q=num_classes)
    input_dict['boxes'] = np.random.rand(1, 5, 2, 4).astype(np.float32)
    input_dict['scores'] = np.random.rand(1, 5, 2).astype(np.float32)
    input_dict['max_output_size_per_class'] = 3
    input_dict['max_total_size'] = 5
    input_dict['name'] = 'class_specific_boxes'
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Padding enabled (pad_per_class=True)
    input_dict['boxes'] = np.random.rand(1, 6, 1, 4).astype(np.float32)
    input_dict['scores'] = np.random.rand(1, 6, 3).astype(np.float32)
    input_dict['max_output_size_per_class'] = 2 # Padded size would be 2 * 3 = 6
    input_dict['max_total_size'] = 10 # Not clipped
    input_dict['pad_per_class'] = True
    input_dict['score_threshold'] = -np.inf
    input_dict['name'] = 'padding_enabled'
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Padding enabled but clipped by max_total_size
    input_dict['max_total_size'] = 5 # Clipped (5 < 2 * 3)
    input_dict['name'] = 'padding_enabled_clipped'
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: No clipping with out-of-bounds coordinates
    input_dict['boxes'] = np.array([[-0.1, -0.1, 1.1, 1.1], [0.5, 0.5, 1.2, 1.2]]).reshape(1, 1, 2, 4).astype(np.float32)
    input_dict['scores'] = np.array([[0.9, 0.8]]).reshape(1, 1, 2).astype(np.float32)
    input_dict['max_output_size_per_class'] = 1
    input_dict['max_total_size'] = 2
    input_dict['iou_threshold'] = 0.5
    input_dict['score_threshold'] = 0.0
    input_dict['pad_per_class'] = False
    input_dict['clip_boxes'] = False
    input_dict['name'] = 'no_clipping'
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batch size > 1
    input_dict['boxes'] = np.random.rand(2, 5, 1, 4).astype(np.float32)
    input_dict['scores'] = np.random.rand(2, 5, 3).astype(np.float32)
    input_dict['max_output_size_per_class'] = 2
    input_dict['max_total_size'] = 5
    input_dict['iou_threshold'] = 0.7
    input_dict['score_threshold'] = 0.3
    input_dict['clip_boxes'] = True
    input_dict['name'] = 'batch_size_2'
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: High suppression with low IOU threshold
    boxes_data = np.array([[0.1, 0.1, 0.5, 0.5], [0.11, 0.11, 0.51, 0.51], [0.8, 0.8, 0.9, 0.9], [0.81, 0.81, 0.91, 0.91]])
    input_dict['boxes'] = boxes_data.reshape(1, 4, 1, 4).astype(np.float32)
    input_dict['scores'] = np.array([[[0.9], [0.8], [0.7], [0.6]]]).astype(np.float32)
    input_dict['max_output_size_per_class'] = 4
    input_dict['max_total_size'] = 4
    input_dict['iou_threshold'] = 0.2
    input_dict['score_threshold'] = 0.1
    input_dict['name'] = 'high_suppression'
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High score threshold
    input_dict['boxes'] = np.random.rand(1, 5, 1, 4).astype(np.float32)
    input_dict['scores'] = np.array([[[0.9, 0.1], [0.8, 0.2], [0.3, 0.3], [0.6, 0.4], [0.4, 0.5]]]).astype(np.float32)
    input_dict['max_output_size_per_class'] = 3
    input_dict['max_total_size'] = 5
    input_dict['iou_threshold'] = 0.5
    input_dict['score_threshold'] = 0.7
    input_dict['name'] = 'high_score_threshold'
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small max_total_size
    input_dict['boxes'] = np.random.rand(1, 10, 5, 4).astype(np.float32)
    input_dict['scores'] = np.random.rand(1, 10, 5).astype(np.float32)
    input_dict['max_output_size_per_class'] = 2
    input_dict['max_total_size'] = 1 # Very small
    input_dict['iou_threshold'] = 0.5
    input_dict['score_threshold'] = 0.0
    input_dict['name'] = 'small_max_total_size'
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative coordinates
    boxes_data = np.array([[-0.2, -0.2, 0.2, 0.2], [-0.8, -0.8, -0.5, -0.5], [-0.1, -0.9, 0.1, -0.7]])
    input_dict['boxes'] = boxes_data.reshape(1, 3, 1, 4).astype(np.float32)
    input_dict['scores'] = np.random.rand(1, 3, 2).astype(np.float32)
    input_dict['max_output_size_per_class'] = 2
    input_dict['max_total_size'] = 3
    input_dict['iou_threshold'] = 0.1
    input_dict['score_threshold'] = -1.0
    input_dict['clip_boxes'] = False
    input_dict['name'] = 'negative_coordinates'
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: IOU threshold of 1.0 (no suppression unless identical)
    boxes_data = np.array([[0.1, 0.1, 0.5, 0.5], [0.1, 0.1, 0.5, 0.5], [0.8, 0.8, 0.9, 0.9], [0.1, 0.2, 0.3, 0.4]])
    input_dict['boxes'] = boxes_data.reshape(1, 4, 1, 4).astype(np.float32)
    input_dict['scores'] = np.array([[[0.9], [0.8], [0.7], [0.6]]]).astype(np.float32)
    input_dict['max_output_size_per_class'] = 4
    input_dict['max_total_size'] = 4
    input_dict['iou_threshold'] = 1.0
    input_dict['score_threshold'] = 0.0
    input_dict['clip_boxes'] = True
    input_dict['name'] = 'iou_threshold_one'
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Zero scores with a positive score_threshold
    input_dict['boxes'] = np.random.rand(2, 5, 3, 4).astype(np.float32)
    input_dict['scores'] = np.zeros((2, 5, 3), dtype=np.float32)
    input_dict['max_output_size_per_class'] = 2
    input_dict['max_total_size'] = 5
    input_dict['iou_threshold'] = 0.5
    input_dict['score_threshold'] = 0.1
    input_dict['name'] = 'zero_scores'
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.combined_non_max_suppression"] = generate_tf_image_combined_non_max_suppression_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.combined_non_max_suppression' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.combined_non_max_suppression'.")

check_valid('tf.image.combined_non_max_suppression', generated_inputs['tf.image.combined_non_max_suppression'], lib="tf", suffix=0)
