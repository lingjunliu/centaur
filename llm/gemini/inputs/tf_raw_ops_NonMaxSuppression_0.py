
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_NonMaxSuppression_inputs():
    list_of_inputs = []

    # Input 1: Basic standard input
    boxes1 = np.array([[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9]], dtype=np.float32)
    scores1 = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size1 = np.array(2, dtype=np.int32)
    iou_threshold1 = 0.5
    name1 = "nms_1"
    list_of_inputs.append({
        'iou_threshold': iou_threshold1,
        'name': name1,
        'boxes': boxes1,
        'scores': scores1,
        'max_output_size': max_output_size1
    })

    # Input 2: Zero overlapping boxes
    boxes2 = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.float32)
    scores2 = np.array([0.5, 0.8], dtype=np.float32)
    max_output_size2 = np.array(5, dtype=np.int32)
    iou_threshold2 = 0.3
    name2 = "nms_2"
    list_of_inputs.append({
        'iou_threshold': iou_threshold2,
        'name': name2,
        'boxes': boxes2,
        'scores': scores2,
        'max_output_size': max_output_size2
    })

    # Input 3: Negative/large coordinates
    boxes3 = np.array([[-10, -10, 10, 10], [-5, -5, 5, 5], [0, 0, 15, 15]], dtype=np.float32)
    scores3 = np.array([-0.1, 0.9, 0.4], dtype=np.float32)
    max_output_size3 = np.array(1, dtype=np.int32)
    iou_threshold3 = 0.1
    name3 = "nms_3"
    list_of_inputs.append({
        'iou_threshold': iou_threshold3,
        'name': name3,
        'boxes': boxes3,
        'scores': scores3,
        'max_output_size': max_output_size3
    })

    # Input 4: Empty boxes
    boxes4 = np.zeros((0, 4), dtype=np.float32)
    scores4 = np.zeros((0,), dtype=np.float32)
    max_output_size4 = np.array(3, dtype=np.int32)
    iou_threshold4 = 0.5
    name4 = "nms_4"
    list_of_inputs.append({
        'iou_threshold': iou_threshold4,
        'name': name4,
        'boxes': boxes4,
        'scores': scores4,
        'max_output_size': max_output_size4
    })

    # Input 5: Max output size is 0
    boxes5 = np.array([[0, 0, 1, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores5 = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size5 = np.array(0, dtype=np.int32)
    iou_threshold5 = 0.5
    name5 = "nms_5"
    list_of_inputs.append({
        'iou_threshold': iou_threshold5,
        'name': name5,
        'boxes': boxes5,
        'scores': scores5,
        'max_output_size': max_output_size5
    })

    # Input 6: Highly overlapping boxes, low iou threshold
    boxes6 = np.array([[0, 0, 10, 10], [1, 1, 9, 9], [2, 2, 8, 8]], dtype=np.float32)
    scores6 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    max_output_size6 = np.array(10, dtype=np.int32)
    iou_threshold6 = 0.05
    name6 = "nms_6"
    list_of_inputs.append({
        'iou_threshold': iou_threshold6,
        'name': name6,
        'boxes': boxes6,
        'scores': scores6,
        'max_output_size': max_output_size6
    })

    # Input 7: iou_threshold is 1.0 (no pruning)
    boxes7 = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
    scores7 = np.array([0.5, 0.5], dtype=np.float32)
    max_output_size7 = np.array(2, dtype=np.int32)
    iou_threshold7 = 1.0
    name7 = "nms_7"
    list_of_inputs.append({
        'iou_threshold': iou_threshold7,
        'name': name7,
        'boxes': boxes7,
        'scores': scores7,
        'max_output_size': max_output_size7
    })

    # Input 8: Many boxes with random coordinates
    np.random.seed(42)
    boxes8 = np.random.rand(100, 4).astype(np.float32)
    for i in range(100):
        if boxes8[i, 2] < boxes8[i, 0]:
            boxes8[i, 0], boxes8[i, 2] = boxes8[i, 2], boxes8[i, 0]
        if boxes8[i, 3] < boxes8[i, 1]:
            boxes8[i, 1], boxes8[i, 3] = boxes8[i, 3], boxes8[i, 1]
    scores8 = np.random.rand(100).astype(np.float32)
    max_output_size8 = np.array(15, dtype=np.int32)
    iou_threshold8 = 0.4
    name8 = "nms_8"
    list_of_inputs.append({
        'iou_threshold': iou_threshold8,
        'name': name8,
        'boxes': boxes8,
        'scores': scores8,
        'max_output_size': max_output_size8
    })

    # Input 9: Identical boxes, varying scores
    boxes9 = np.array([[0, 0, 1, 1]] * 10, dtype=np.float32)
    scores9 = np.array([float(i) for i in range(10)], dtype=np.float32)
    max_output_size9 = np.array(5, dtype=np.int32)
    iou_threshold9 = 0.5
    name9 = "nms_9"
    list_of_inputs.append({
        'iou_threshold': iou_threshold9,
        'name': name9,
        'boxes': boxes9,
        'scores': scores9,
        'max_output_size': max_output_size9
    })

    # Input 10: Non-overlapping sequential boxes
    boxes10 = np.array([[i, i, i + 0.5, i + 0.5] for i in range(10)], dtype=np.float32)
    scores10 = np.array([0.1 * i for i in range(10)], dtype=np.float32)
    max_output_size10 = np.array(100, dtype=np.int32)
    iou_threshold10 = 0.5
    name10 = "nms_10"
    list_of_inputs.append({
        'iou_threshold': iou_threshold10,
        'name': name10,
        'boxes': boxes10,
        'scores': scores10,
        'max_output_size': max_output_size10
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.NonMaxSuppression"] = tf_raw_ops_NonMaxSuppression_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.NonMaxSuppression' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NonMaxSuppression'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.NonMaxSuppression', generated_inputs['tf.raw_ops.NonMaxSuppression'], lib="tf", suffix=0)
