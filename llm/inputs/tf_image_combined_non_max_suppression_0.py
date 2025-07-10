
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_combined_non_max_suppression_inputs():
    list_of_inputs = []

    # Input 1
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.5, 0.5, 0.7, 0.7]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(1)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.6)
    pad_per_class = False
    clip_boxes = True
    name = "nms1"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.5, 0.5, 0.7, 0.7]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.int32(2)
    max_total_size = np.int32(2)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.6)
    pad_per_class = True
    clip_boxes = False
    name = "nms2"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.5, 0.5, 0.7, 0.7]]],
                      [[[0.2, 0.2, 0.4, 0.4]], [[0.6, 0.6, 0.8, 0.8]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]], [[0.9, 0.6]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(2)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.5)
    pad_per_class = False
    clip_boxes = True
    name = "nms3"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.5, 0.5, 0.7, 0.7]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(2)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(-1.0)
    pad_per_class = False
    clip_boxes = True
    name = "nms4"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.5, 0.5, 0.7, 0.7]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.int32(2)
    max_total_size = np.int32(1)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.6)
    pad_per_class = False
    clip_boxes = True
    name = "nms5"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.5, 0.5, 0.7, 0.7]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(2)
    iou_threshold = np.float32(0.3)
    score_threshold = np.float32(0.5)
    pad_per_class = False
    clip_boxes = True
    name = "nms6"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.5, 0.5, 0.7, 0.7]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(1)
    iou_threshold = np.float32(0.9)
    score_threshold = np.float32(0.6)
    pad_per_class = False
    clip_boxes = True
    name = "nms7"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.5, 0.5, 0.7, 0.7]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(1)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.6)
    pad_per_class = False
    clip_boxes = True
    name = None
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.5, 0.5, 0.7, 0.7]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(1)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.6)
    pad_per_class = True
    clip_boxes = False
    name = "nms9"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.5, 0.5, 0.7, 0.7]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(1)
    iou_threshold = np.float32(0.1)
    score_threshold = np.float32(0.9)
    pad_per_class = False
    clip_boxes = True
    name = "nms10"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11. Corrected scores shape to match boxes
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.5, 0.5, 0.7, 0.7]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(5)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.1)
    pad_per_class = True
    clip_boxes = True
    name = "nms11"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Multiple classes
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.5, 0.5, 0.7, 0.7]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(2)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.3)
    pad_per_class = False
    clip_boxes = True
    name = "nms12"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: batch_size=2, num_boxes=3, num_classes=2
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]], [[0.5, 0.5, 0.7, 0.7], [0.6, 0.6, 0.8, 0.8]], [[0.7, 0.7, 0.9, 0.9], [0.8, 0.8, 1.0, 1.0]]],
                    [[[0.15, 0.15, 0.35, 0.35], [0.25, 0.25, 0.45, 0.45]], [[0.55, 0.55, 0.75, 0.75], [0.65, 0.65, 0.85, 0.85]], [[0.75, 0.75, 0.95, 0.95], [0.85, 0.85, 1.05, 1.05]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7], [0.6, 0.5], [0.9, 0.4]], [[0.7, 0.6], [0.5, 0.4], [0.8, 0.3]]], dtype=np.float32)
    max_output_size_per_class = np.int32(2)
    max_total_size = np.int32(4)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.4)
    pad_per_class = False
    clip_boxes = True
    name = "nms13"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14: Corrected scores shape to match boxes, multiple classes
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.5, 0.5, 0.7, 0.7]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(2)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.3)
    pad_per_class = False
    clip_boxes = True
    name = "nms14"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 15
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]], [[0.5, 0.5, 0.7, 0.7], [0.6, 0.6, 0.8, 0.8]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7], [0.6, 0.5]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(2)
    iou_threshold = np.float32(0.3)
    score_threshold = np.float32(0.5)
    pad_per_class = False
    clip_boxes = True
    name = "nms15"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 16
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]], [[0.5, 0.5, 0.7, 0.7], [0.6, 0.6, 0.8, 0.8]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7], [0.6, 0.5]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(4)
    iou_threshold = np.float32(0.3)
    score_threshold = np.float32(0.1)
    pad_per_class = True
    clip_boxes = True
    name = "nms16"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 17: Correct score dimensions
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]], [[0.5, 0.5, 0.7, 0.7], [0.6, 0.6, 0.8, 0.8]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7], [0.6, 0.5]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(4)
    iou_threshold = np.float32(0.3)
    score_threshold = np.float32(0.1)
    pad_per_class = True
    clip_boxes = True
    name = "nms17"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 18: One box, two classes
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(2)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.6)
    pad_per_class = False
    clip_boxes = True
    name = "nms18"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 19: shape of scores corrected
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.5, 0.5, 0.7, 0.7]]]], dtype=np.float32)
    scores = np.array([[[0.8, 0.7]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(2)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.3)
    pad_per_class = False
    clip_boxes = True
    name = "nms19"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 20: batch_size=1, num_boxes=2, num_classes=1
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.5, 0.5, 0.7, 0.7]]]], dtype=np.float32)
    scores = np.array([[[0.8], [0.7]]]], dtype=np.float32)
    max_output_size_per_class = np.int32(1)
    max_total_size = np.int32(2)
    iou_threshold = np.float32(0.5)
    score_threshold = np.float32(0.3)
    pad_per_class = False
    clip_boxes = True
    name = "nms20"
    input_dict = {'boxes': boxes, 'scores': scores, 'max_output_size_per_class': max_output_size_per_class,
                  'max_total_size': max_total_size, 'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold, 'pad_per_class': pad_per_class, 'clip_boxes': clip_boxes,
                  'name': name}
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
