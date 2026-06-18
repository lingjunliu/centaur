
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_sample_distorted_bounding_box_inputs():
    list_of_inputs = []

    # Input 1: Standard values
    input_dict_1 = {
        'image_size': np.array([224, 224, 3], dtype=np.int32),
        'bounding_boxes': np.array([[[0.1, 0.1, 0.9, 0.9]]], dtype=np.float32),
        'seed': 42,
        'min_object_covered': 0.1,
        'aspect_ratio_range': [0.75, 1.33],
        'area_range': [0.05, 1.0],
        'max_attempts': 100,
        'use_image_if_no_bounding_boxes': False,
        'name': "sample_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Larger image and multiple bounding boxes
    input_dict_2 = {
        'image_size': np.array([512, 512, 3], dtype=np.int32),
        'bounding_boxes': np.array([[[0.1, 0.1, 0.5, 0.5], [0.5, 0.5, 0.9, 0.9]]], dtype=np.float32),
        'seed': 10,
        'min_object_covered': 0.5,
        'aspect_ratio_range': [0.5, 2.0],
        'area_range': [0.1, 0.9],
        'max_attempts': 50,
        'use_image_if_no_bounding_boxes': True,
        'name': "sample_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: int64 image_size, min_object_covered=0.0, non-zero seed
    input_dict_3 = {
        'image_size': np.array([256, 256, 1], dtype=np.int64),
        'bounding_boxes': np.array([[[0.2, 0.2, 0.8, 0.8]]], dtype=np.float32),
        'seed': 3,
        'min_object_covered': 0.0,
        'aspect_ratio_range': [0.8, 1.2],
        'area_range': [0.2, 0.8],
        'max_attempts': 200,
        'use_image_if_no_bounding_boxes': False,
        'name': "sample_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: uint8 image_size, empty bounding_boxes, use_image_if_no_bounding_boxes=True
    input_dict_4 = {
        'image_size': np.array([64, 64, 3], dtype=np.uint8),
        'bounding_boxes': np.zeros((1, 0, 4), dtype=np.float32),
        'seed': 1234,
        'min_object_covered': 0.1,
        'aspect_ratio_range': [0.75, 1.33],
        'area_range': [0.05, 1.0],
        'max_attempts': 100,
        'use_image_if_no_bounding_boxes': True,
        'name': "sample_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Different aspect ratio range
    input_dict_5 = {
        'image_size': np.array([480, 640, 3], dtype=np.int32),
        'bounding_boxes': np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32),
        'seed': 7,
        'min_object_covered': 0.3,
        'aspect_ratio_range': [1.0, 1.5],
        'area_range': [0.3, 0.7],
        'max_attempts': 80,
        'use_image_if_no_bounding_boxes': False,
        'name': "sample_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Multiple bounding boxes with tight area range
    input_dict_6 = {
        'image_size': np.array([128, 128, 4], dtype=np.int32),
        'bounding_boxes': np.array([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.1, 0.1, 0.9, 0.9]]], dtype=np.float32),
        'seed': 99,
        'min_object_covered': 0.9,
        'aspect_ratio_range': [0.9, 1.1],
        'area_range': [0.8, 1.0],
        'max_attempts': 150,
        'use_image_if_no_bounding_boxes': False,
        'name': "sample_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: int16 image_size
    input_dict_7 = {
        'image_size': np.array([32, 32, 3], dtype=np.int16),
        'bounding_boxes': np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32),
        'seed': 43,
        'min_object_covered': 0.2,
        'aspect_ratio_range': [0.5, 1.5],
        'area_range': [0.1, 0.5],
        'max_attempts': 10,
        'use_image_if_no_bounding_boxes': True,
        'name': "sample_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: int8 image_size
    input_dict_8 = {
        'image_size': np.array([16, 16, 1], dtype=np.int8),
        'bounding_boxes': np.array([[[0.0, 0.0, 0.5, 0.5]]], dtype=np.float32),
        'seed': 1,
        'min_object_covered': 0.05,
        'aspect_ratio_range': [0.6, 1.4],
        'area_range': [0.4, 0.9],
        'max_attempts': 300,
        'use_image_if_no_bounding_boxes': False,
        'name': "sample_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: High max_attempts, wide range
    input_dict_9 = {
        'image_size': np.array([1000, 1000, 3], dtype=np.int32),
        'bounding_boxes': np.array([[[0.15, 0.25, 0.75, 0.85]]], dtype=np.float32),
        'seed': 999,
        'min_object_covered': 0.45,
        'aspect_ratio_range': [0.1, 10.0],
        'area_range': [0.01, 1.0],
        'max_attempts': 500,
        'use_image_if_no_bounding_boxes': True,
        'name': "sample_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Square aspect ratio constraint
    input_dict_10 = {
        'image_size': np.array([224, 224, 3], dtype=np.int32),
        'bounding_boxes': np.array([[[0.2, 0.2, 0.8, 0.8]]], dtype=np.float32),
        'seed': 123,
        'min_object_covered': 0.8,
        'aspect_ratio_range': [1.0, 1.0],
        'area_range': [0.5, 0.5],
        'max_attempts': 20,
        'use_image_if_no_bounding_boxes': False,
        'name': "sample_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.image.sample_distorted_bounding_box"] = tf_image_sample_distorted_bounding_box_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.sample_distorted_bounding_box' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.sample_distorted_bounding_box'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.sample_distorted_bounding_box', generated_inputs['tf.image.sample_distorted_bounding_box'], lib="tf", suffix=0)
