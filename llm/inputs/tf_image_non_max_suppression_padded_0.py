
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_non_max_suppression_padded_inputs():
    list_of_inputs = []

    # Input 1
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = 0.4
    pad_to_max_output_size = True
    name = "nms_padded_1"
    sorted_input = False
    canonicalized_coordinates = False
    tile_size = 512
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size, 'iou_threshold': iou_threshold, 'score_threshold': score_threshold, 'pad_to_max_output_size': pad_to_max_output_size, 'name': name, 'sorted_input': sorted_input, 'canonicalized_coordinates': canonicalized_coordinates, 'tile_size': tile_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2 - Batch input
    boxes = np.array([[[0, 0, 1, 1], [0, 0.5, 1, 1.5]], [[0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]]], dtype=np.float32)
    scores = np.array([[0.9, 0.75], [0.6, 0.95]], dtype=np.float32)
    max_output_size = 2
    iou_threshold = 0.5
    score_threshold = 0.4
    pad_to_max_output_size = True
    name = "nms_padded_2"
    sorted_input = False
    canonicalized_coordinates = False
    tile_size = 512
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size, 'iou_threshold': iou_threshold, 'score_threshold': score_threshold, 'pad_to_max_output_size': pad_to_max_output_size, 'name': name, 'sorted_input': sorted_input, 'canonicalized_coordinates': canonicalized_coordinates, 'tile_size': tile_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 - Different iou threshold
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.9
    score_threshold = 0.4
    pad_to_max_output_size = True
    name = "nms_padded_3"
    sorted_input = False
    canonicalized_coordinates = False
    tile_size = 512
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size, 'iou_threshold': iou_threshold, 'score_threshold': score_threshold, 'pad_to_max_output_size': pad_to_max_output_size, 'name': name, 'sorted_input': sorted_input, 'canonicalized_coordinates': canonicalized_coordinates, 'tile_size': tile_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 - Different score threshold
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = 0.8
    pad_to_max_output_size = True
    name = "nms_padded_4"
    sorted_input = False
    canonicalized_coordinates = False
    tile_size = 512
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size, 'iou_threshold': iou_threshold, 'score_threshold': score_threshold, 'pad_to_max_output_size': pad_to_max_output_size, 'name': name, 'sorted_input': sorted_input, 'canonicalized_coordinates': canonicalized_coordinates, 'tile_size': tile_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - Sorted input
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.95, 0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = 0.4
    pad_to_max_output_size = True
    name = "nms_padded_5"
    sorted_input = True
    canonicalized_coordinates = False
    tile_size = 512
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size, 'iou_threshold': iou_threshold, 'score_threshold': score_threshold, 'pad_to_max_output_size': pad_to_max_output_size, 'name': name, 'sorted_input': sorted_input, 'canonicalized_coordinates': canonicalized_coordinates, 'tile_size': tile_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - Different tile size
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = 0.4
    pad_to_max_output_size = True
    name = "nms_padded_7"
    sorted_input = False
    canonicalized_coordinates = False
    tile_size = 128
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size, 'iou_threshold': iou_threshold, 'score_threshold': score_threshold, 'pad_to_max_output_size': pad_to_max_output_size, 'name': name, 'sorted_input': sorted_input, 'canonicalized_coordinates': canonicalized_coordinates, 'tile_size': tile_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Negative coordinates
    boxes = np.array([[-1, -1, 0, 0], [0, 0, 1, 1]], dtype=np.float32)
    scores = np.array([0.9, 0.75], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = 0.4
    pad_to_max_output_size = True
    name = "nms_padded_9"
    sorted_input = False
    canonicalized_coordinates = False
    tile_size = 512
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size, 'iou_threshold': iou_threshold, 'score_threshold': score_threshold, 'pad_to_max_output_size': pad_to_max_output_size, 'name': name, 'sorted_input': sorted_input, 'canonicalized_coordinates': canonicalized_coordinates, 'tile_size': tile_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - Very high score
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5]], dtype=np.float32)
    scores = np.array([100000000000.0, 0.75], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = 0.4
    pad_to_max_output_size = True
    name = "nms_padded_10"
    sorted_input = False
    canonicalized_coordinates = False
    tile_size = 512
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size, 'iou_threshold': iou_threshold, 'score_threshold': score_threshold, 'pad_to_max_output_size': pad_to_max_output_size, 'name': name, 'sorted_input': sorted_input, 'canonicalized_coordinates': canonicalized_coordinates, 'tile_size': tile_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - Zero iou and score threshold
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.0
    score_threshold = 0.0
    pad_to_max_output_size = True
    name = "nms_padded_11"
    sorted_input = False
    canonicalized_coordinates = False
    tile_size = 512
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size, 'iou_threshold': iou_threshold, 'score_threshold': score_threshold, 'pad_to_max_output_size': pad_to_max_output_size, 'name': name, 'sorted_input': sorted_input, 'canonicalized_coordinates': canonicalized_coordinates, 'tile_size': tile_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - Negative score threshold
    boxes = np.array([[0, 0, 1, 1], [0, 0.5, 1, 1.5], [0.5, 0, 1.5, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6, 0.95], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = -1.0
    pad_to_max_output_size = True
    name = "nms_padded_12"
    sorted_input = False
    canonicalized_coordinates = False
    tile_size = 512
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size, 'iou_threshold': iou_threshold, 'score_threshold': score_threshold, 'pad_to_max_output_size': pad_to_max_output_size, 'name': name, 'sorted_input': sorted_input, 'canonicalized_coordinates': canonicalized_coordinates, 'tile_size': tile_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.non_max_suppression_padded"] = tf_image_non_max_suppression_padded_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.non_max_suppression_padded' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.non_max_suppression_padded'.")

check_valid('tf.image.non_max_suppression_padded', generated_inputs['tf.image.non_max_suppression_padded'], lib="tf", suffix=0)
