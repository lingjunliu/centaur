
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_generate_bounding_box_proposals_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    scores = np.random.rand(1, 10, 10, 3).astype(np.float32)
    bbox_deltas = np.random.rand(1, 10, 10, 12).astype(np.float32)
    image_info = np.array([[600, 800, 1.0, 600, 800]]).astype(np.float32)
    anchors = np.random.rand(3, 4).astype(np.float32)
    nms_threshold = np.float32(0.5)
    pre_nms_topn = np.int32(1000)
    min_size = np.float32(8.0)
    post_nms_topn = np.int32(100)
    name = None

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
    scores = np.random.rand(2, 5, 5, 5).astype(np.float32)
    bbox_deltas = np.random.rand(2, 5, 5, 20).astype(np.float32)
    image_info = np.array([[300, 400, 0.5, 300, 400], [600, 800, 1.0, 600, 800]]).astype(np.float32)
    anchors = np.random.rand(5, 4).astype(np.float32)
    nms_threshold = np.float32(0.6)
    pre_nms_topn = np.int32(2000)
    min_size = np.float32(10.0)
    post_nms_topn = np.int32(200)
    name = None
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

    # Input 3: Different anchor format
    scores = np.random.rand(1, 8, 8, 4).astype(np.float32)
    bbox_deltas = np.random.rand(1, 8, 8, 16).astype(np.float32)
    image_info = np.array([[400, 500, 0.8, 400, 500]]).astype(np.float32)
    anchors = np.random.rand(4, 4).astype(np.float32)
    nms_threshold = np.float32(0.4)
    pre_nms_topn = np.int32(3000)
    min_size = np.float32(12.0)
    post_nms_topn = np.int32(300)
    name = None

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

     # Input 4: Smaller image size
    scores = np.random.rand(1, 3, 3, 2).astype(np.float32)
    bbox_deltas = np.random.rand(1, 3, 3, 8).astype(np.float32)
    image_info = np.array([[150, 200, 1.0, 150, 200]]).astype(np.float32)
    anchors = np.random.rand(2, 4).astype(np.float32)
    nms_threshold = np.float32(0.3)
    pre_nms_topn = np.int32(500)
    min_size = np.float32(5.0)
    post_nms_topn = np.int32(50)
    name = None
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

    # Input 5: different image sizes, larger anchors
    scores = np.random.rand(2, 7, 7, 3).astype(np.float32)
    bbox_deltas = np.random.rand(2, 7, 7, 12).astype(np.float32)
    image_info = np.array([[500, 600, 0.9, 500, 600], [700, 900, 1.1, 700, 900]]).astype(np.float32)
    anchors = np.random.rand(3, 4).astype(np.float32) * 100
    nms_threshold = np.float32(0.7)
    pre_nms_topn = np.int32(4000)
    min_size = np.float32(20.0)
    post_nms_topn = np.int32(400)
    name = None
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

    # Input 6: Minimal sizes, fewer anchors
    scores = np.random.rand(1, 2, 2, 1).astype(np.float32)
    bbox_deltas = np.random.rand(1, 2, 2, 4).astype(np.float32)
    image_info = np.array([[100, 120, 1.0, 100, 120]]).astype(np.float32)
    anchors = np.random.rand(1, 4).astype(np.float32)
    nms_threshold = np.float32(0.2)
    pre_nms_topn = np.int32(100)
    min_size = np.float32(2.0)
    post_nms_topn = np.int32(10)
    name = None
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

    # Input 7: High resolution image
    scores = np.random.rand(1, 20, 20, 6).astype(np.float32)
    bbox_deltas = np.random.rand(1, 20, 20, 24).astype(np.float32)
    image_info = np.array([[1200, 1600, 1.0, 1200, 1600]]).astype(np.float32)
    anchors = np.random.rand(6, 4).astype(np.float32) * 200
    nms_threshold = np.float32(0.8)
    pre_nms_topn = np.int32(7000)
    min_size = np.float32(32.0)
    post_nms_topn = np.int32(500)
    name = None

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

    # Input 8: Rectangular image, different scales
    scores = np.random.rand(2, 4, 6, 2).astype(np.float32)
    bbox_deltas = np.random.rand(2, 4, 6, 8).astype(np.float32)
    image_info = np.array([[250, 350, 0.6, 250, 350], [450, 550, 0.7, 450, 550]]).astype(np.float32)
    anchors = np.random.rand(2, 4).astype(np.float32)
    nms_threshold = np.float32(0.3)
    pre_nms_topn = np.int32(800)
    min_size = np.float32(7.0)
    post_nms_topn = np.int32(80)
    name = None

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

    # Input 9: Small sizes
    scores = np.random.rand(1, 1, 1, 1).astype(np.float32)
    bbox_deltas = np.random.rand(1, 1, 1, 4).astype(np.float32)
    image_info = np.array([[64, 64, 1.0, 64, 64]]).astype(np.float32)
    anchors = np.random.rand(1, 4).astype(np.float32)
    nms_threshold = np.float32(0.1)
    pre_nms_topn = np.int32(50)
    min_size = np.float32(1.0)
    post_nms_topn = np.int32(5)
    name = None
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

    # Input 10: Different aspect ratios
    scores = np.random.rand(1, 6, 8, 3).astype(np.float32)
    bbox_deltas = np.random.rand(1, 6, 8, 12).astype(np.float32)
    image_info = np.array([[300, 500, 1.0, 300, 500]]).astype(np.float32)
    anchors = np.random.rand(3, 4).astype(np.float32) * 50
    nms_threshold = np.float32(0.9)
    pre_nms_topn = np.int32(5000)
    min_size = np.float32(24.0)
    post_nms_topn = np.int32(600)
    name = None
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

generated_inputs = {}
generated_inputs["tf.image.generate_bounding_box_proposals"] = tf_image_generate_bounding_box_proposals_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.generate_bounding_box_proposals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.generate_bounding_box_proposals'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.generate_bounding_box_proposals', generated_inputs['tf.image.generate_bounding_box_proposals'], lib="tf", suffix=0)
