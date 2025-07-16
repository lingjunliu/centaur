
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np

def tf_image_generate_bounding_box_proposals_inputs():
    list_of_inputs = []

    # Input 1
    scores = np.random.rand(1, 10, 10, 3).astype(np.float32)
    bbox_deltas = np.random.rand(1, 10, 10, 12).astype(np.float32)
    image_info = np.array([[600, 800, 1.0, 600, 800]]).astype(np.float32)
    anchors = np.random.rand(1, 3, 4).astype(np.float32)
    nms_threshold = 0.7
    pre_nms_topn = 6000
    min_size = 16.0
    post_nms_topn = 300
    name = "proposal_generator"

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
    list_of_inputs.append(input_dict)

    # Input 2
    scores = np.random.rand(2, 5, 5, 2).astype(np.float32)
    bbox_deltas = np.random.rand(2, 5, 5, 8).astype(np.float32)
    image_info = np.array([[300, 400, 0.5, 300, 400], [300, 400, 0.5, 300, 400]]).astype(np.float32)
    anchors = np.random.rand(1, 2, 4).astype(np.float32)
    nms_threshold = 0.5
    pre_nms_topn = 3000
    min_size = 8.0
    post_nms_topn = 150
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
    list_of_inputs.append(input_dict)

    # Input 3
    scores = np.random.rand(1, 20, 20, 4).astype(np.float32)
    bbox_deltas = np.random.rand(1, 20, 20, 16).astype(np.float32)
    image_info = np.array([[1200, 1600, 2.0, 1200, 1600]]).astype(np.float32)
    anchors = np.random.rand(1, 4, 4).astype(np.float32)
    nms_threshold = 0.9
    pre_nms_topn = 10000
    min_size = 32.0
    post_nms_topn = 500
    name = "another_proposal"

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
    list_of_inputs.append(input_dict)

    # Input 4
    scores = np.random.rand(3, 8, 8, 1).astype(np.float32)
    bbox_deltas = np.random.rand(3, 8, 8, 4).astype(np.float32)
    image_info = np.array([[400, 600, 1.0, 400, 600], [400, 600, 1.0, 400, 600], [400, 600, 1.0, 400, 600]]).astype(np.float32)
    anchors = np.random.rand(1, 1, 4).astype(np.float32)
    nms_threshold = 0.6
    pre_nms_topn = 4000
    min_size = 12.0
    post_nms_topn = 200
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
    list_of_inputs.append(input_dict)

     # Input 5
    scores = np.random.rand(1, 4, 4, 5).astype(np.float32)
    bbox_deltas = np.random.rand(1, 4, 4, 20).astype(np.float32)
    image_info = np.array([[200, 300, 0.8, 200, 300]]).astype(np.float32)
    anchors = np.random.rand(1, 5, 4).astype(np.float32)
    nms_threshold = 0.4
    pre_nms_topn = 2000
    min_size = 4.0
    post_nms_topn = 100
    name = "small_proposal"

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
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.generate_bounding_box_proposals"] = tf_image_generate_bounding_box_proposals_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.generate_bounding_box_proposals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.generate_bounding_box_proposals'.")

check_valid('tf.image.generate_bounding_box_proposals', generated_inputs['tf.image.generate_bounding_box_proposals'], lib="tf", suffix=0)
