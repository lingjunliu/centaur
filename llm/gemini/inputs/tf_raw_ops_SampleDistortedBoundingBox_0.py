
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SampleDistortedBoundingBox_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "seed": 12,
        "seed2": 34,
        "min_object_covered": 0.1,
        "aspect_ratio_range": [0.75, 1.33],
        "area_range": [0.05, 1.0],
        "max_attempts": 100,
        "use_image_if_no_bounding_boxes": False,
        "name": "sample_1",
        "image_size": np.array([256, 256, 3], dtype=np.int32),
        "bounding_boxes": np.array([[[0.1, 0.1, 0.9, 0.9]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "seed": 42,
        "seed2": 42,
        "min_object_covered": 0.5,
        "aspect_ratio_range": [0.5, 2.0],
        "area_range": [0.1, 0.9],
        "max_attempts": 50,
        "use_image_if_no_bounding_boxes": True,
        "name": "sample_2",
        "image_size": np.array([512, 512, 1], dtype=np.int64),
        "bounding_boxes": np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "seed": 1,
        "seed2": 2,
        "min_object_covered": 0.25,
        "aspect_ratio_range": [0.8, 1.25],
        "area_range": [0.2, 0.8],
        "max_attempts": 200,
        "use_image_if_no_bounding_boxes": False,
        "name": "sample_3",
        "image_size": np.array([100, 100, 4], dtype=np.int16),
        "bounding_boxes": np.array([[[0.2, 0.2, 0.8, 0.8], [0.1, 0.1, 0.5, 0.5]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "seed": 123,
        "seed2": 456,
        "min_object_covered": 0.0,
        "aspect_ratio_range": [0.9, 1.1],
        "area_range": [0.3, 0.7],
        "max_attempts": 10,
        "use_image_if_no_bounding_boxes": True,
        "name": "sample_4",
        "image_size": np.array([64, 64, 3], dtype=np.int32),
        "bounding_boxes": np.array([[[0.2, 0.2, 0.8, 0.8]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "seed": 7,
        "seed2": 8,
        "min_object_covered": 0.3,
        "aspect_ratio_range": [0.6, 1.6],
        "area_range": [0.15, 0.85],
        "max_attempts": 150,
        "use_image_if_no_bounding_boxes": False,
        "name": "sample_5",
        "image_size": np.array([32, 32, 3], dtype=np.uint8),
        "bounding_boxes": np.array([[[0.0, 0.0, 0.5, 0.5]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "seed": 10,
        "seed2": 20,
        "min_object_covered": 0.0,
        "aspect_ratio_range": [0.5, 1.5],
        "area_range": [0.4, 0.9],
        "max_attempts": 20,
        "use_image_if_no_bounding_boxes": True,
        "name": "sample_6",
        "image_size": np.array([120, 120, 3], dtype=np.int8),
        "bounding_boxes": np.array([[[0.15, 0.15, 0.85, 0.85]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "seed": 100,
        "seed2": 200,
        "min_object_covered": 0.75,
        "aspect_ratio_range": [0.3, 3.0],
        "area_range": [0.01, 1.0],
        "max_attempts": 500,
        "use_image_if_no_bounding_boxes": False,
        "name": "sample_7",
        "image_size": np.array([480, 640, 3], dtype=np.int32),
        "bounding_boxes": np.array([[[0.05, 0.05, 0.95, 0.95]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "seed": 99,
        "seed2": 99,
        "min_object_covered": 0.9,
        "aspect_ratio_range": [1.0, 1.0],
        "area_range": [0.05, 0.1],
        "max_attempts": 300,
        "use_image_if_no_bounding_boxes": True,
        "name": "sample_8",
        "image_size": np.array([1000, 1000, 3], dtype=np.int32),
        "bounding_boxes": np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "seed": 1234,
        "seed2": 5678,
        "min_object_covered": 0.05,
        "aspect_ratio_range": [0.1, 10.0],
        "area_range": [0.05, 1.0],
        "max_attempts": 1000,
        "use_image_if_no_bounding_boxes": False,
        "name": "sample_9",
        "image_size": np.array([16, 16, 3], dtype=np.int32),
        "bounding_boxes": np.array([[[0.1, 0.1, 0.9, 0.9]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "seed": 111,
        "seed2": 222,
        "min_object_covered": 0.4,
        "aspect_ratio_range": [0.7, 1.4],
        "area_range": [0.1, 0.9],
        "max_attempts": 80,
        "use_image_if_no_bounding_boxes": False,
        "name": "sample_10",
        "image_size": np.array([1920, 1080, 3], dtype=np.int32),
        "bounding_boxes": np.array([[[0.3, 0.3, 0.7, 0.7]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SampleDistortedBoundingBox"] = tf_raw_ops_SampleDistortedBoundingBox_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SampleDistortedBoundingBox' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SampleDistortedBoundingBox'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SampleDistortedBoundingBox', generated_inputs['tf.raw_ops.SampleDistortedBoundingBox'], lib="tf", suffix=0)
