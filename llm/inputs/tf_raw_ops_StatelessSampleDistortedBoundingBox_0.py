
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_stateless_sample_distorted_bounding_box_inputs():
    list_of_inputs = []

    # Input 1
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    min_object_covered = np.array(0.5, dtype=np.float32)
    seed = np.array([123, 456], dtype=np.int32)
    aspect_ratio_range = [0.5, 2.0]
    area_range = [0.1, 0.9]
    max_attempts = 50
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_1"

    input_dict = {
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name,
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image_size = np.array([128, 128, 1], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.1, 0.9, 0.9], [0.3, 0.3, 0.6, 0.6]]], dtype=np.float32)
    min_object_covered = np.array(0.2, dtype=np.float32)
    seed = np.array([789, 101], dtype=np.int32)
    aspect_ratio_range = [0.8, 1.2]
    area_range = [0.2, 0.8]
    max_attempts = 100
    use_image_if_no_bounding_boxes = True
    name = "distorted_bbox_2"

    input_dict = {
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name,
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image_size = np.array([64, 64, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    min_object_covered = np.array(0.0, dtype=np.float32)
    seed = np.array([112, 131], dtype=np.int32)
    aspect_ratio_range = [0.7, 1.4]
    area_range = [0.05, 1.0]
    max_attempts = 200
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_3"

    input_dict = {
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name,
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image_size = np.array([512, 512, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.3, 0.2, 0.8, 0.9]]], dtype=np.float32)
    min_object_covered = np.array(0.7, dtype=np.float32)
    seed = np.array([141, 151], dtype=np.int32)
    aspect_ratio_range = [0.6, 1.5]
    area_range = [0.3, 0.7]
    max_attempts = 75
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_4"

    input_dict = {
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name,
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    image_size = np.array([32, 32, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.2, 0.3, 0.4]]], dtype=np.float32)
    min_object_covered = np.array(0.9, dtype=np.float32)
    seed = np.array([161, 171], dtype=np.int32)
    aspect_ratio_range = [0.9, 1.1]
    area_range = [0.4, 0.6]
    max_attempts = 125
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_5"

    input_dict = {
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name,
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    min_object_covered = np.array(0.5, dtype=np.float32)
    seed = np.array([123, 456], dtype=np.int32)
    aspect_ratio_range = [0.5, 2.0]
    area_range = [0.1, 0.9]
    max_attempts = 50
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_6"

    input_dict = {
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name,
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    image_size = np.array([128, 128, 1], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.1, 0.9, 0.9], [0.3, 0.3, 0.6, 0.6]]], dtype=np.float32)
    min_object_covered = np.array(0.2, dtype=np.float32)
    seed = np.array([789, 101], dtype=np.int32)
    aspect_ratio_range = [0.8, 1.2]
    area_range = [0.2, 0.8]
    max_attempts = 100
    use_image_if_no_bounding_boxes = True
    name = "distorted_bbox_7"

    input_dict = {
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name,
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    image_size = np.array([64, 64, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    min_object_covered = np.array(0.0, dtype=np.float32)
    seed = np.array([112, 131], dtype=np.int32)
    aspect_ratio_range = [0.7, 1.4]
    area_range = [0.05, 1.0]
    max_attempts = 200
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_8"

    input_dict = {
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name,
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    image_size = np.array([512, 512, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.3, 0.2, 0.8, 0.9]]], dtype=np.float32)
    min_object_covered = np.array(0.7, dtype=np.float32)
    seed = np.array([141, 151], dtype=np.int32)
    aspect_ratio_range = [0.6, 1.5]
    area_range = [0.3, 0.7]
    max_attempts = 75
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_9"

    input_dict = {
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name,
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    image_size = np.array([32, 32, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.2, 0.3, 0.4]]], dtype=np.float32)
    min_object_covered = np.array(0.9, dtype=np.float32)
    seed = np.array([161, 171], dtype=np.int32)
    aspect_ratio_range = [0.9, 1.1]
    area_range = [0.4, 0.6]
    max_attempts = 125
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_10"

    input_dict = {
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name,
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "min_object_covered": min_object_covered,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StatelessSampleDistortedBoundingBox"] = tf_raw_ops_stateless_sample_distorted_bounding_box_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StatelessSampleDistortedBoundingBox' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StatelessSampleDistortedBoundingBox'.")

check_valid('tf.raw_ops.StatelessSampleDistortedBoundingBox', generated_inputs['tf.raw_ops.StatelessSampleDistortedBoundingBox'], lib="tf", suffix=0)
