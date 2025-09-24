
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sample_distorted_bounding_box_v2_inputs():
    list_of_inputs = []

    # Input 1
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    min_object_covered = np.array(0.5, dtype=np.float32)
    seed = 123
    seed2 = 456
    aspect_ratio_range = [0.5, 2.0]
    area_range = [0.1, 0.8]
    max_attempts = 50
    use_image_if_no_bounding_boxes = False
    name = "distorted_bounding_box"

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

    # Input 2
    image_size = np.array([128, 128, 1], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.1, 0.9, 0.9], [0.3, 0.3, 0.6, 0.6]]], dtype=np.float32)
    min_object_covered = np.array(0.2, dtype=np.float32)
    seed = 789
    seed2 = 101
    aspect_ratio_range = [0.8, 1.2]
    area_range = [0.2, 0.9]
    max_attempts = 100
    use_image_if_no_bounding_boxes = True
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

    # Input 3
    image_size = np.array([64, 64, 3], dtype=np.int32)
    bounding_boxes = np.array([[]], dtype=np.float32).reshape((1,0,4))
    min_object_covered = np.array(0.0, dtype=np.float32)
    seed = 10
    seed2 = 20
    aspect_ratio_range = [0.75, 1.33]
    area_range = [0.05, 1.0]
    max_attempts = 100
    use_image_if_no_bounding_boxes = True
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

    # Input 4
    image_size = np.array([512, 512, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    min_object_covered = np.array(1.0, dtype=np.float32)
    seed = 1024
    seed2 = 2048
    aspect_ratio_range = [0.9, 1.1]
    area_range = [0.7, 1.0]
    max_attempts = 20
    use_image_if_no_bounding_boxes = False
    name = "full_image"

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

    # Input 5
    image_size = np.array([32, 32, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.2, 0.3, 0.5, 0.6]]], dtype=np.float32)
    min_object_covered = np.array(0.8, dtype=np.float32)
    seed = 1
    seed2 = 2
    aspect_ratio_range = [0.6, 1.5]
    area_range = [0.3, 0.7]
    max_attempts = 75
    use_image_if_no_bounding_boxes = False
    name = "partial_crop"

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
    
    # Input 6
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    min_object_covered = np.array(0.5, dtype=np.float32)
    seed = 123
    seed2 = 456
    aspect_ratio_range = [0.5, 2.0]
    area_range = [0.1, 0.8]
    max_attempts = 50
    use_image_if_no_bounding_boxes = False
    name = "distorted_bounding_box"

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
    
    # Input 7
    image_size = np.array([640, 480, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]], dtype=np.float32)
    min_object_covered = np.array(0.3, dtype=np.float32)
    seed = 42
    seed2 = 420
    aspect_ratio_range = [0.66, 1.5]
    area_range = [0.15, 0.95]
    max_attempts = 80
    use_image_if_no_bounding_boxes = False
    name = "complex_scene"

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
    
    # Input 8
    image_size = np.array([128, 128, 4], dtype=np.int32)
    bounding_boxes = np.array([[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]], dtype=np.float32)
    min_object_covered = np.array(0.6, dtype=np.float32)
    seed = 999
    seed2 = 111
    aspect_ratio_range = [0.9, 1.1]
    area_range = [0.6, 0.9]
    max_attempts = 60
    use_image_if_no_bounding_boxes = False
    name = "two_objects"

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
    
    # Input 9
    image_size = np.array([256, 256, 1], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.1, 0.9, 0.9]]], dtype=np.float32)
    min_object_covered = np.array(0.9, dtype=np.float32)
    seed = 1234
    seed2 = 5678
    aspect_ratio_range = [0.7, 1.4]
    area_range = [0.4, 0.8]
    max_attempts = 90
    use_image_if_no_bounding_boxes = False
    name = "single_object"

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

    # Input 10
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    min_object_covered = np.array(0.0, dtype=np.float32)
    seed = 123
    seed2 = 456
    aspect_ratio_range = [0.1, 10.0]
    area_range = [0.01, 0.99]
    max_attempts = 50
    use_image_if_no_bounding_boxes = True
    name = "distorted_bounding_box"

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
    
    # Input 11
    image_size = np.array([100, 150, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.3, 0.4, 0.8, 0.9]]], dtype=np.float32)
    min_object_covered = np.array(0.7, dtype=np.float32)
    seed = 500
    seed2 = 600
    aspect_ratio_range = [0.85, 1.15]
    area_range = [0.5, 0.75]
    max_attempts = 30
    use_image_if_no_bounding_boxes = False
    name = "specific_values"

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
generated_inputs["tf.raw_ops.SampleDistortedBoundingBoxV2"] = tf_raw_ops_sample_distorted_bounding_box_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SampleDistortedBoundingBoxV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SampleDistortedBoundingBoxV2'.")

check_valid('tf.raw_ops.SampleDistortedBoundingBoxV2', generated_inputs['tf.raw_ops.SampleDistortedBoundingBoxV2'], lib="tf", suffix=0)
