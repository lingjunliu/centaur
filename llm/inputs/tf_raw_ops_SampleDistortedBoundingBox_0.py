
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sampledistortedboundingbox_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": 123,
        "seed2": 456,
        "min_object_covered": 0.5,
        "aspect_ratio_range": [0.5, 2.0],
        "area_range": [0.2, 0.8],
        "max_attempts": 50,
        "use_image_if_no_bounding_boxes": False,
        "name": "distorted_bbox_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: No bounding boxes, use_image_if_no_bounding_boxes = True
    image_size = np.array([128, 128, 1], dtype=np.int32)
    bounding_boxes = np.array([[]], dtype=np.float32).reshape(1, 0, 4)
    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": 789,
        "seed2": 101,
        "min_object_covered": 0.0,
        "aspect_ratio_range": [0.75, 1.33],
        "area_range": [0.05, 1.0],
        "max_attempts": 100,
        "use_image_if_no_bounding_boxes": True,
        "name": "distorted_bbox_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple bounding boxes
    image_size = np.array([512, 512, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.9, 0.9]]], dtype=np.float32)
    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": 202,
        "seed2": 303,
        "min_object_covered": 0.2,
        "aspect_ratio_range": [0.6, 1.5],
        "area_range": [0.1, 0.9],
        "max_attempts": 75,
        "use_image_if_no_bounding_boxes": False,
        "name": "distorted_bbox_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4: Different image size type
    image_size = np.array([64, 64, 3], dtype=np.uint8)
    bounding_boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": 404,
        "seed2": 505,
        "min_object_covered": 0.9,
        "aspect_ratio_range": [1.0, 1.0],
        "area_range": [0.9, 1.0],
        "max_attempts": 10,
        "use_image_if_no_bounding_boxes": False,
        "name": "distorted_bbox_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  min_object_covered = 0
    image_size = np.array([100, 150, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.2, 0.3, 0.5, 0.6]]], dtype=np.float32)

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": 606,
        "seed2": 707,
        "min_object_covered": 0.0,
        "aspect_ratio_range": [0.8, 1.2],
        "area_range": [0.3, 0.7],
        "max_attempts": 20,
        "use_image_if_no_bounding_boxes": False,
        "name": "distorted_bbox_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Aspect ratio range is a single value
    image_size = np.array([200, 300, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.4, 0.5, 0.7, 0.8]]], dtype=np.float32)
    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": 808,
        "seed2": 909,
        "min_object_covered": 0.3,
        "aspect_ratio_range": [1.0, 1.0],
        "area_range": [0.4, 0.6],
        "max_attempts": 30,
        "use_image_if_no_bounding_boxes": False,
        "name": "distorted_bbox_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: area range is close to [0,0]
    image_size = np.array([50, 50, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.1, 0.9, 0.9]]], dtype=np.float32)
    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": 1010,
        "seed2": 1111,
        "min_object_covered": 0.9,
        "aspect_ratio_range": [0.8, 1.2],
        "area_range": [0.01, 0.02],
        "max_attempts": 20,
        "use_image_if_no_bounding_boxes": False,
        "name": "distorted_bbox_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64 image size
    image_size = np.array([256, 256, 3], dtype=np.int64)
    bounding_boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": 123,
        "seed2": 456,
        "min_object_covered": 0.5,
        "aspect_ratio_range": [0.5, 2.0],
        "area_range": [0.2, 0.8],
        "max_attempts": 50,
        "use_image_if_no_bounding_boxes": False,
        "name": "distorted_bbox_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: seed = 0
    image_size = np.array([128, 128, 1], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.1, 0.9, 0.9]]], dtype=np.float32)
    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": 0,
        "seed2": 0,
        "min_object_covered": 0.0,
        "aspect_ratio_range": [0.75, 1.33],
        "area_range": [0.05, 1.0],
        "max_attempts": 100,
        "use_image_if_no_bounding_boxes": False,
        "name": "distorted_bbox_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multiple batches
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.25, 0.25, 0.75, 0.75]], [[0.1, 0.1, 0.4, 0.4]]], dtype=np.float32)
    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": 123,
        "seed2": 456,
        "min_object_covered": 0.5,
        "aspect_ratio_range": [0.5, 2.0],
        "area_range": [0.2, 0.8],
        "max_attempts": 50,
        "use_image_if_no_bounding_boxes": False,
        "name": "distorted_bbox_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SampleDistortedBoundingBox"] = tf_raw_ops_sampledistortedboundingbox_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SampleDistortedBoundingBox' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SampleDistortedBoundingBox'.")

check_valid('tf.raw_ops.SampleDistortedBoundingBox', generated_inputs['tf.raw_ops.SampleDistortedBoundingBox'], lib="tf", suffix=0)
