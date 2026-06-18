
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_non_max_suppression_overlaps_inputs():
    list_of_inputs = []
    
    # Input 1
    overlaps1 = np.array([[1.0, 0.4, 0.3], [0.4, 1.0, 0.2], [0.3, 0.2, 1.0]], dtype=np.float32)
    scores1 = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    input_dict1 = {
        'overlaps': overlaps1,
        'scores': scores1,
        'max_output_size': int(2),
        'overlap_threshold': float(0.5),
        'score_threshold': float(0.1),
        'name': "nms_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    overlaps2 = np.array([[1.0]], dtype=np.float32)
    scores2 = np.array([0.5], dtype=np.float32)
    input_dict2 = {
        'overlaps': overlaps2,
        'scores': scores2,
        'max_output_size': int(1),
        'overlap_threshold': float(0.3),
        'score_threshold': float(0.0),
        'name': "nms_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    overlaps3 = np.eye(10, dtype=np.float32)
    scores3 = np.linspace(0.1, 1.0, 10, dtype=np.float32)
    input_dict3 = {
        'overlaps': overlaps3,
        'scores': scores3,
        'max_output_size': int(5),
        'overlap_threshold': float(0.5),
        'score_threshold': float(0.5),
        'name': "nms_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    overlaps4 = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.4], [0.2, 0.4, 1.0]], dtype=np.float32)
    scores4 = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    input_dict4 = {
        'overlaps': overlaps4,
        'scores': scores4,
        'max_output_size': int(2),
        'overlap_threshold': float(0.7),
        'score_threshold': float(0.0),
        'name': "nms_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    overlaps5 = np.eye(4, dtype=np.float32)
    scores5 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    input_dict5 = {
        'overlaps': overlaps5,
        'scores': scores5,
        'max_output_size': int(10),
        'overlap_threshold': float(0.1),
        'score_threshold': float(-1.0),
        'name': "nms_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    overlaps6 = np.ones((6, 6), dtype=np.float32)
    scores6 = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32)
    input_dict6 = {
        'overlaps': overlaps6,
        'scores': scores6,
        'max_output_size': int(1),
        'overlap_threshold': float(0.99),
        'score_threshold': float(0.8),
        'name': "nms_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    temp = np.arange(16, dtype=np.float32).reshape(4, 4)
    overlaps7 = (temp + temp.T) / 32.0
    np.fill_diagonal(overlaps7, 1.0)
    scores7 = np.array([0.9, 0.1, 0.2, 0.8], dtype=np.float32)
    input_dict7 = {
        'overlaps': overlaps7,
        'scores': scores7,
        'max_output_size': int(3),
        'overlap_threshold': float(0.45),
        'score_threshold': float(0.2),
        'name': "nms_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    overlaps8 = np.eye(2, dtype=np.float32)
    scores8 = np.array([-0.1, -0.2], dtype=np.float32)
    input_dict8 = {
        'overlaps': overlaps8,
        'scores': scores8,
        'max_output_size': int(2),
        'overlap_threshold': float(0.5),
        'score_threshold': float(-0.5),
        'name': "nms_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    overlaps9 = np.eye(5, dtype=np.float32) * 0.9
    scores9 = np.array([0.95, 0.85, 0.75, 0.65, 0.55], dtype=np.float32)
    input_dict9 = {
        'overlaps': overlaps9,
        'scores': scores9,
        'max_output_size': int(0),
        'overlap_threshold': float(0.5),
        'score_threshold': float(0.0),
        'name': "nms_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    overlaps10 = np.array([[1.0, 0.9, 0.0, 0.0], [0.9, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.9], [0.0, 0.0, 0.9, 1.0]], dtype=np.float32)
    scores10 = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    input_dict10 = {
        'overlaps': overlaps10,
        'scores': scores10,
        'max_output_size': int(2),
        'overlap_threshold': float(0.8),
        'score_threshold': float(0.5),
        'name': "nms_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.non_max_suppression_overlaps"] = tf_image_non_max_suppression_overlaps_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.non_max_suppression_overlaps' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.non_max_suppression_overlaps'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.non_max_suppression_overlaps', generated_inputs['tf.image.non_max_suppression_overlaps'], lib="tf", suffix=0)
