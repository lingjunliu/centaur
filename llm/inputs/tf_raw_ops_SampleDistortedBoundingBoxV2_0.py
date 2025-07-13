
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SampleDistortedBoundingBoxV2_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.2, 0.3, 0.8, 0.9]]], dtype=np.float32)
    min_object_covered = np.array(0.5, dtype=np.float32)
    seed = 123
    seed2 = 456
    aspect_ratio_range = [0.75, 1.33]
    area_range = [0.05, 1.0]
    max_attempts = 100
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed,
        "seed2": seed2,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple bounding boxes
    image_size = np.array([512, 512, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]], dtype=np.float32)
    min_object_covered = np.array(0.2, dtype=np.float32)
    seed = 789
    seed2 = 101
    aspect_ratio_range = [0.5, 2.0]
    area_range = [0.1, 0.8]
    max_attempts = 50
    use_image_if_no_bounding_boxes = False
    name = None

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed,
        "seed2": seed2,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different image size type
    image_size = np.array([640, 480, 1], dtype=np.int64)
    bounding_boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    min_object_covered = np.array(0.8, dtype=np.float32)
    seed = 0
    seed2 = 0
    aspect_ratio_range = [0.9, 1.1]
    area_range = [0.5, 0.9]
    max_attempts = 200
    use_image_if_no_bounding_boxes = False
    name = "all_image"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed,
        "seed2": seed2,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: No bounding boxes, use_image = True
    image_size = np.array([128, 128, 3], dtype=np.int32)
    bounding_boxes = np.array([[]], dtype=np.float32)  # Empty bounding boxes
    min_object_covered = np.array(0.0, dtype=np.float32)
    seed = 42
    seed2 = 24
    aspect_ratio_range = [0.6, 1.5]
    area_range = [0.2, 0.7]
    max_attempts = 30
    use_image_if_no_bounding_boxes = True
    name = "no_boxes"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed,
        "seed2": seed2,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: min_object_covered = 0
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.2, 0.3, 0.8, 0.9]]], dtype=np.float32)
    min_object_covered = np.array(0.0, dtype=np.float32)
    seed = 123
    seed2 = 456
    aspect_ratio_range = [0.75, 1.33]
    area_range = [0.05, 1.0]
    max_attempts = 100
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed,
        "seed2": seed2,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: Different image size type (uint8)
    image_size = np.array([64, 64, 3], dtype=np.uint8)
    bounding_boxes = np.array([[[0.1, 0.1, 0.9, 0.9]]], dtype=np.float32)
    min_object_covered = np.array(0.3, dtype=np.float32)
    seed = 1
    seed2 = 2
    aspect_ratio_range = [0.8, 1.2]
    area_range = [0.3, 0.6]
    max_attempts = 60
    use_image_if_no_bounding_boxes = False
    name = "small_image"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed,
        "seed2": seed2,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different image size type (int16)
    image_size = np.array([256, 256, 3], dtype=np.int16)
    bounding_boxes = np.array([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]], dtype=np.float32)
    min_object_covered = np.array(0.1, dtype=np.float32)
    seed = 5
    seed2 = 6
    aspect_ratio_range = [0.6, 1.6]
    area_range = [0.01, 0.9]
    max_attempts = 120
    use_image_if_no_bounding_boxes = False
    name = "int16_image"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed,
        "seed2": seed2,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multiple batches
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.2, 0.3, 0.8, 0.9]], [[0.1, 0.2, 0.7, 0.8]]], dtype=np.float32) # (2, 1, 4)
    min_object_covered = np.array(0.6, dtype=np.float32)
    seed = 7
    seed2 = 8
    aspect_ratio_range = [0.9, 1.1]
    area_range = [0.4, 0.7]
    max_attempts = 90
    use_image_if_no_bounding_boxes = False
    name = "multiple_batches"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed,
        "seed2": seed2,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: very small image
    image_size = np.array([10, 10, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    min_object_covered = np.array(0.01, dtype=np.float32)
    seed = 1234
    seed2 = 4321
    aspect_ratio_range = [0.75, 1.33]
    area_range = [0.01, 1.0]
    max_attempts = 100
    use_image_if_no_bounding_boxes = False
    name = "small_image_test"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed,
        "seed2": seed2,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SampleDistortedBoundingBoxV2"] = tf_raw_ops_SampleDistortedBoundingBoxV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SampleDistortedBoundingBoxV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SampleDistortedBoundingBoxV2'.")

check_valid('tf.raw_ops.SampleDistortedBoundingBoxV2', generated_inputs['tf.raw_ops.SampleDistortedBoundingBoxV2'], lib="tf", suffix=0)
