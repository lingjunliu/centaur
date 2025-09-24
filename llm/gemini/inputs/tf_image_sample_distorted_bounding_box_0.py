
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_sample_distorted_bounding_box_inputs():
    list_of_inputs = []

    # Input 1
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    seed = 123
    min_object_covered = 0.5
    aspect_ratio_range = [0.5, 2.0]
    area_range = [0.2, 0.8]
    max_attempts = 50
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_1"
    input_dict = {"image_size": image_size, "bounding_boxes": bounding_boxes, "seed": seed,
                  "min_object_covered": min_object_covered, "aspect_ratio_range": aspect_ratio_range,
                  "area_range": area_range, "max_attempts": max_attempts,
                  "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image_size = np.array([128, 128, 1], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.1, 0.9, 0.9]]], dtype=np.float32)
    seed = 456
    min_object_covered = 0.9
    aspect_ratio_range = [0.9, 1.1]
    area_range = [0.7, 1.0]
    max_attempts = 10
    use_image_if_no_bounding_boxes = True
    name = "distorted_bbox_2"
    input_dict = {"image_size": image_size, "bounding_boxes": bounding_boxes, "seed": seed,
                  "min_object_covered": min_object_covered, "aspect_ratio_range": aspect_ratio_range,
                  "area_range": area_range, "max_attempts": max_attempts,
                  "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image_size = np.array([64, 64, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    seed = 789
    min_object_covered = 0.0
    aspect_ratio_range = [0.1, 10.0]
    area_range = [0.01, 0.99]
    max_attempts = 200
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_3"
    input_dict = {"image_size": image_size, "bounding_boxes": bounding_boxes, "seed": seed,
                  "min_object_covered": min_object_covered, "aspect_ratio_range": aspect_ratio_range,
                  "area_range": area_range, "max_attempts": max_attempts,
                  "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image_size = np.array([512, 512, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]], dtype=np.float32)
    seed = 101
    min_object_covered = 0.2
    aspect_ratio_range = [0.6, 1.5]
    area_range = [0.1, 0.5]
    max_attempts = 75
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_4"
    input_dict = {"image_size": image_size, "bounding_boxes": bounding_boxes, "seed": seed,
                  "min_object_covered": min_object_covered, "aspect_ratio_range": aspect_ratio_range,
                  "area_range": area_range, "max_attempts": max_attempts,
                  "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    image_size = np.array([32, 32, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]], dtype=np.float32)
    seed = 202
    min_object_covered = 0.3
    aspect_ratio_range = [0.8, 1.25]
    area_range = [0.3, 0.7]
    max_attempts = 125
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_5"
    input_dict = {"image_size": image_size, "bounding_boxes": bounding_boxes, "seed": seed,
                  "min_object_covered": min_object_covered, "aspect_ratio_range": aspect_ratio_range,
                  "area_range": area_range, "max_attempts": max_attempts,
                  "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    seed = 1 # Changed to non-zero
    min_object_covered = 0.0
    aspect_ratio_range = [0.75, 1.33] # Provide default
    area_range = [0.05, 1] # Provide default
    max_attempts = 100
    use_image_if_no_bounding_boxes = True
    name = "distorted_bbox_6"
    input_dict = {"image_size": image_size, "bounding_boxes": bounding_boxes, "seed": seed,
                  "min_object_covered": min_object_covered, "aspect_ratio_range": aspect_ratio_range,
                  "area_range": area_range, "max_attempts": max_attempts,
                  "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    image_size = np.array([128, 128, 1], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.1, 0.9, 0.9]]], dtype=np.float32)
    seed = 1
    min_object_covered = 0.01
    aspect_ratio_range = [0.75, 1.33]
    area_range = [0.05, 1]
    max_attempts = 100
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_7"
    input_dict = {"image_size": image_size, "bounding_boxes": bounding_boxes, "seed": seed,
                  "min_object_covered": min_object_covered, "aspect_ratio_range": aspect_ratio_range,
                  "area_range": area_range, "max_attempts": max_attempts,
                  "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    image_size = np.array([64, 64, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    seed = 2
    min_object_covered = 0.99
    aspect_ratio_range = [0.76, 1.34]
    area_range = [0.06, 0.99]
    max_attempts = 101
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_8"
    input_dict = {"image_size": image_size, "bounding_boxes": bounding_boxes, "seed": seed,
                  "min_object_covered": min_object_covered, "aspect_ratio_range": aspect_ratio_range,
                  "area_range": area_range, "max_attempts": max_attempts,
                  "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    image_size = np.array([512, 512, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]], dtype=np.float32)
    seed = 3
    min_object_covered = 0.5
    aspect_ratio_range = [0.74, 1.32]
    area_range = [0.04, 0.98]
    max_attempts = 99
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_9"
    input_dict = {"image_size": image_size, "bounding_boxes": bounding_boxes, "seed": seed,
                  "min_object_covered": min_object_covered, "aspect_ratio_range": aspect_ratio_range,
                  "area_range": area_range, "max_attempts": max_attempts,
                  "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    image_size = np.array([32, 32, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]], dtype=np.float32)
    seed = 4
    min_object_covered = 0.7
    aspect_ratio_range = [0.73, 1.31]
    area_range = [0.03, 0.97]
    max_attempts = 98
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_10"
    input_dict = {"image_size": image_size, "bounding_boxes": bounding_boxes, "seed": seed,
                  "min_object_covered": min_object_covered, "aspect_ratio_range": aspect_ratio_range,
                  "area_range": area_range, "max_attempts": max_attempts,
                  "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.sample_distorted_bounding_box"] = tf_image_sample_distorted_bounding_box_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.sample_distorted_bounding_box' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.sample_distorted_bounding_box'.")

check_valid('tf.image.sample_distorted_bounding_box', generated_inputs['tf.image.sample_distorted_bounding_box'], lib="tf", suffix=0)
