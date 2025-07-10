
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_sample_distorted_bounding_box_inputs():
    list_of_inputs = []

    # Input 1
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    seed = np.array([123, 456], dtype=np.int32)
    min_object_covered = 0.2
    aspect_ratio_range = [0.5, 2.0]
    area_range = [0.1, 0.9]
    max_attempts = 50
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_1"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": seed,
        "min_object_covered": min_object_covered,
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
    seed = np.array([789, 101], dtype=np.int32)
    min_object_covered = 0.5
    aspect_ratio_range = [0.8, 1.25]
    area_range = [0.2, 0.8]
    max_attempts = 100
    use_image_if_no_bounding_boxes = True
    name = "distorted_bbox_2"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": seed,
        "min_object_covered": min_object_covered,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image_size = np.array([64, 64, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    seed = np.array([112, 131], dtype=np.int32)
    min_object_covered = 0.0
    aspect_ratio_range = [0.75, 1.33]
    area_range = [0.05, 1.0]
    max_attempts = 200
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_3"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": seed,
        "min_object_covered": min_object_covered,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4
    image_size = np.array([512, 512, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.2, 0.3, 0.8, 0.9]]], dtype=np.float32)
    seed = np.array([222, 333], dtype=np.int32)
    min_object_covered = 0.7
    aspect_ratio_range = [0.6, 1.5]
    area_range = [0.3, 0.7]
    max_attempts = 75
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_4"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": seed,
        "min_object_covered": min_object_covered,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    image_size = np.array([32, 32, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.1, 0.2, 0.4, 0.5], [0.6, 0.7, 0.9, 0.95]]], dtype=np.float32)
    seed = np.array([444, 555], dtype=np.int32)
    min_object_covered = 0.9
    aspect_ratio_range = [0.9, 1.1]
    area_range = [0.4, 0.6]
    max_attempts = 25
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_5"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": seed,
        "min_object_covered": min_object_covered,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - empty bounding boxes
    image_size = np.array([128, 128, 3], dtype=np.int32)
    bounding_boxes = np.array([[]], dtype=np.float32).reshape(1, 0, 4)
    seed = np.array([555, 666], dtype=np.int32)
    min_object_covered = 0.1
    aspect_ratio_range = [0.75, 1.33]
    area_range = [0.05, 1.0]
    max_attempts = 100
    use_image_if_no_bounding_boxes = True
    name = "distorted_bbox_6"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": seed,
        "min_object_covered": min_object_covered,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.0, 0.0, 0.5, 0.5]]], dtype=np.float32)
    seed = np.array([123, 789], dtype=np.int32)
    min_object_covered = 0.2
    aspect_ratio_range = [0.5, 2.0]
    area_range = [0.1, 0.9]
    max_attempts = 50
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_7"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": seed,
        "min_object_covered": min_object_covered,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    seed = np.array([987, 654], dtype=np.int32)
    min_object_covered = 0.2
    aspect_ratio_range = [0.5, 2.0]
    area_range = [0.1, 0.9]
    max_attempts = 50
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_8"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": seed,
        "min_object_covered": min_object_covered,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    image_size = np.array([256, 256, 3], dtype=np.int32)
    bounding_boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    seed = np.array([321, 456], dtype=np.int32)
    min_object_covered = 0.2
    aspect_ratio_range = [0.5, 2.0]
    area_range = [0.1, 0.9]
    max_attempts = 50
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_9"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": seed,
        "min_object_covered": min_object_covered,
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
    seed = np.array([123, 654], dtype=np.int32)
    min_object_covered = 0.2
    aspect_ratio_range = [0.5, 2.0]
    area_range = [0.1, 0.9]
    max_attempts = 50
    use_image_if_no_bounding_boxes = False
    name = "distorted_bbox_10"

    input_dict = {
        "image_size": image_size,
        "bounding_boxes": bounding_boxes,
        "seed": seed,
        "min_object_covered": min_object_covered,
        "aspect_ratio_range": aspect_ratio_range,
        "area_range": area_range,
        "max_attempts": max_attempts,
        "use_image_if_no_bounding_boxes": use_image_if_no_bounding_boxes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.stateless_sample_distorted_bounding_box"] = tf_image_stateless_sample_distorted_bounding_box_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.stateless_sample_distorted_bounding_box' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_sample_distorted_bounding_box'.")

check_valid('tf.image.stateless_sample_distorted_bounding_box', generated_inputs['tf.image.stateless_sample_distorted_bounding_box'], lib="tf", suffix=0)
