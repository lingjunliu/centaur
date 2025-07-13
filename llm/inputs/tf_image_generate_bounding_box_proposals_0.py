
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_generate_bounding_box_proposals_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    scores = np.random.rand(1, 10, 10, 3).astype(np.float32)
    bbox_deltas = np.random.rand(1, 10, 10, 12).astype(np.float32)
    image_info = np.array([[600, 800, 1.0, 600, 800]], dtype=np.float32)
    anchors = np.random.rand(1, 3, 4).astype(np.float32)  # Shape (1, num_anchors, 4)
    nms_threshold = 0.7
    pre_nms_topn = 6000
    min_size = 16.0
    post_nms_topn = 300
    name = "basic_input"

    input_dict = {
        "scores": scores,
        "bbox_deltas": bbox_deltas,
        "image_info": image_info,
        "anchors": anchors,
        "nms_threshold": nms_threshold,
        "pre_nms_topn": pre_nms_topn,
        "min_size": min_size,
        "post_nms_topn": post_nms_topn,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple images
    scores = np.random.rand(2, 10, 10, 3).astype(np.float32)
    bbox_deltas = np.random.rand(2, 10, 10, 12).astype(np.float32)
    image_info = np.array([[600, 800, 1.0, 600, 800], [800, 600, 1.0, 800, 600]], dtype=np.float32)
    anchors = np.random.rand(1, 3, 4).astype(np.float32)  # Shape (1, num_anchors, 4)
    nms_threshold = 0.6
    pre_nms_topn = 5000
    min_size = 15.0
    post_nms_topn = 250
    name = "multiple_images"

    input_dict = {
        "scores": scores,
        "bbox_deltas": bbox_deltas,
        "image_info": image_info,
        "anchors": anchors,
        "nms_threshold": nms_threshold,
        "pre_nms_topn": pre_nms_topn,
        "min_size": min_size,
        "post_nms_topn": post_nms_topn,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different image size
    scores = np.random.rand(1, 20, 20, 5).astype(np.float32)
    bbox_deltas = np.random.rand(1, 20, 20, 20).astype(np.float32)
    image_info = np.array([[800, 1000, 1.0, 800, 1000]], dtype=np.float32)
    anchors = np.random.rand(1, 5, 4).astype(np.float32)  # Shape (1, num_anchors, 4)
    nms_threshold = 0.8
    pre_nms_topn = 7000
    min_size = 17.0
    post_nms_topn = 350
    name = "different_image_size"

    input_dict = {
        "scores": scores,
        "bbox_deltas": bbox_deltas,
        "image_info": image_info,
        "anchors": anchors,
        "nms_threshold": nms_threshold,
        "pre_nms_topn": pre_nms_topn,
        "min_size": min_size,
        "post_nms_topn": post_nms_topn,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different anchor count
    scores = np.random.rand(1, 10, 10, 7).astype(np.float32)
    bbox_deltas = np.random.rand(1, 10, 10, 28).astype(np.float32)
    image_info = np.array([[600, 800, 1.0, 600, 800]], dtype=np.float32)
    anchors = np.random.rand(1, 7, 4).astype(np.float32)  # Shape (1, num_anchors, 4)
    nms_threshold = 0.75
    pre_nms_topn = 6500
    min_size = 16.5
    post_nms_topn = 325
    name = "different_anchor_count"

    input_dict = {
        "scores": scores,
        "bbox_deltas": bbox_deltas,
        "image_info": image_info,
        "anchors": anchors,
        "nms_threshold": nms_threshold,
        "pre_nms_topn": pre_nms_topn,
        "min_size": min_size,
        "post_nms_topn": post_nms_topn,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small values
    scores = np.random.rand(1, 5, 5, 2).astype(np.float32) * 0.1
    bbox_deltas = np.random.rand(1, 5, 5, 8).astype(np.float32) * 0.1
    image_info = np.array([[300, 400, 1.0, 300, 400]], dtype=np.float32)
    anchors = np.random.rand(1, 2, 4).astype(np.float32)  # Shape (1, num_anchors, 4)
    nms_threshold = 0.5
    pre_nms_topn = 1000
    min_size = 8.0
    post_nms_topn = 50
    name = "small_values"

    input_dict = {
        "scores": scores,
        "bbox_deltas": bbox_deltas,
        "image_info": image_info,
        "anchors": anchors,
        "nms_threshold": nms_threshold,
        "pre_nms_topn": pre_nms_topn,
        "min_size": min_size,
        "post_nms_topn": post_nms_topn,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger values
    scores = np.random.rand(1, 15, 15, 4).astype(np.float32) * 10
    bbox_deltas = np.random.rand(1, 15, 15, 16).astype(np.float32) * 10
    image_info = np.array([[1200, 1600, 1.0, 1200, 1600]], dtype=np.float32)
    anchors = np.random.rand(1, 4, 4).astype(np.float32)  # Shape (1, num_anchors, 4)
    nms_threshold = 0.9
    pre_nms_topn = 8000
    min_size = 32.0
    post_nms_topn = 400
    name = "larger_values"

    input_dict = {
        "scores": scores,
        "bbox_deltas": bbox_deltas,
        "image_info": image_info,
        "anchors": anchors,
        "nms_threshold": nms_threshold,
        "pre_nms_topn": pre_nms_topn,
        "min_size": min_size,
        "post_nms_topn": post_nms_topn,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different scale
    scores = np.random.rand(1, 10, 10, 3).astype(np.float32)
    bbox_deltas = np.random.rand(1, 10, 10, 12).astype(np.float32)
    image_info = np.array([[600, 800, 0.5, 600, 800]], dtype=np.float32)
    anchors = np.random.rand(1, 3, 4).astype(np.float32)  # Shape (1, num_anchors, 4)
    nms_threshold = 0.7
    pre_nms_topn = 6000
    min_size = 16.0
    post_nms_topn = 300
    name = "different_scale"

    input_dict = {
        "scores": scores,
        "bbox_deltas": bbox_deltas,
        "image_info": image_info,
        "anchors": anchors,
        "nms_threshold": nms_threshold,
        "pre_nms_topn": pre_nms_topn,
        "min_size": min_size,
        "post_nms_topn": post_nms_topn,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: Very small min_size
    scores = np.random.rand(1, 10, 10, 3).astype(np.float32)
    bbox_deltas = np.random.rand(1, 10, 10, 12).astype(np.float32)
    image_info = np.array([[600, 800, 1.0, 600, 800]], dtype=np.float32)
    anchors = np.random.rand(1, 3, 4).astype(np.float32)  # Shape (1, num_anchors, 4)
    nms_threshold = 0.7
    pre_nms_topn = 6000
    min_size = 1.0
    post_nms_topn = 300
    name = "very_small_min_size"

    input_dict = {
        "scores": scores,
        "bbox_deltas": bbox_deltas,
        "image_info": image_info,
        "anchors": anchors,
        "nms_threshold": nms_threshold,
        "pre_nms_topn": pre_nms_topn,
        "min_size": min_size,
        "post_nms_topn": post_nms_topn,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very Large min_size
    scores = np.random.rand(1, 10, 10, 3).astype(np.float32)
    bbox_deltas = np.random.rand(1, 10, 10, 12).astype(np.float32)
    image_info = np.array([[600, 800, 1.0, 600, 800]], dtype=np.float32)
    anchors = np.random.rand(1, 3, 4).astype(np.float32)  # Shape (1, num_anchors, 4)
    nms_threshold = 0.7
    pre_nms_topn = 6000
    min_size = 100.0
    post_nms_topn = 300
    name = "very_large_min_size"

    input_dict = {
        "scores": scores,
        "bbox_deltas": bbox_deltas,
        "image_info": image_info,
        "anchors": anchors,
        "nms_threshold": nms_threshold,
        "pre_nms_topn": pre_nms_topn,
        "min_size": min_size,
        "post_nms_topn": post_nms_topn,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different post_nms_topn
    scores = np.random.rand(1, 10, 10, 3).astype(np.float32)
    bbox_deltas = np.random.rand(1, 10, 10, 12).astype(np.float32)
    image_info = np.array([[600, 800, 1.0, 600, 800]], dtype=np.float32)
    anchors = np.random.rand(1, 3, 4).astype(np.float32)  # Shape (1, num_anchors, 4)
    nms_threshold = 0.7
    pre_nms_topn = 6000
    min_size = 16.0
    post_nms_topn = 100
    name = "different_post_nms_topn"

    input_dict = {
        "scores": scores,
        "bbox_deltas": bbox_deltas,
        "image_info": image_info,
        "anchors": anchors,
        "nms_threshold": nms_threshold,
        "pre_nms_topn": pre_nms_topn,
        "min_size": min_size,
        "post_nms_topn": post_nms_topn,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.generate_bounding_box_proposals"] = tf_image_generate_bounding_box_proposals_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.generate_bounding_box_proposals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.generate_bounding_box_proposals'.")

check_valid('tf.image.generate_bounding_box_proposals', generated_inputs['tf.image.generate_bounding_box_proposals'], lib="tf", suffix=0)
