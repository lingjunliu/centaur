
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_generate_bounding_box_proposals_inputs():
    list_of_inputs = []

    # Input 1
    scores = np.random.rand(1, 10, 10, 3).astype(np.float32)
    bbox_deltas = np.random.rand(1, 10, 10, 12).astype(np.float32)
    image_info = np.array([[600, 800, 1.0, 600, 800]], dtype=np.float32)
    anchors = np.random.rand(1, 3, 4).astype(np.float32)
    nms_threshold = 0.7
    pre_nms_topn = 6000
    min_size = 16.0
    post_nms_topn = 300
    name = "proposal_1"

    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': nms_threshold,
        'pre_nms_topn': pre_nms_topn,
        'min_size': min_size,
        'post_nms_topn': post_nms_topn,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    scores = np.random.rand(2, 5, 5, 2).astype(np.float32)
    bbox_deltas = np.random.rand(2, 5, 5, 8).astype(np.float32)
    image_info = np.array([[300, 400, 0.5, 300, 400],[300, 400, 0.5, 300, 400]], dtype=np.float32)
    anchors = np.random.rand(1, 2, 4).astype(np.float32)
    nms_threshold = 0.5
    pre_nms_topn = 3000
    min_size = 8.0
    post_nms_topn = 150
    name = "proposal_2"

    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': nms_threshold,
        'pre_nms_topn': pre_nms_topn,
        'min_size': min_size,
        'post_nms_topn': post_nms_topn,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    scores = np.random.rand(1, 20, 20, 5).astype(np.float32)
    bbox_deltas = np.random.rand(1, 20, 20, 20).astype(np.float32)
    image_info = np.array([[1200, 1600, 2.0, 1200, 1600]], dtype=np.float32)
    anchors = np.random.rand(1, 5, 4).astype(np.float32)
    nms_threshold = 0.9
    pre_nms_topn = 8000
    min_size = 32.0
    post_nms_topn = 400
    name = "proposal_3"

    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': nms_threshold,
        'pre_nms_topn': pre_nms_topn,
        'min_size': min_size,
        'post_nms_topn': post_nms_topn,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    scores = np.random.rand(3, 3, 3, 1).astype(np.float32)
    bbox_deltas = np.random.rand(3, 3, 3, 4).astype(np.float32)
    image_info = np.array([[150, 200, 0.25, 150, 200],[150, 200, 0.25, 150, 200],[150, 200, 0.25, 150, 200]], dtype=np.float32)
    anchors = np.random.rand(1, 1, 4).astype(np.float32)
    nms_threshold = 0.3
    pre_nms_topn = 1000
    min_size = 4.0
    post_nms_topn = 50
    name = "proposal_4"

    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': nms_threshold,
        'pre_nms_topn': pre_nms_topn,
        'min_size': min_size,
        'post_nms_topn': post_nms_topn,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    scores = np.random.rand(1, 1, 1, 1).astype(np.float32)
    bbox_deltas = np.random.rand(1, 1, 1, 4).astype(np.float32)
    image_info = np.array([[50, 50, 1.0, 50, 50]], dtype=np.float32)
    anchors = np.random.rand(1, 1, 4).astype(np.float32)
    nms_threshold = 0.1
    pre_nms_topn = 10
    min_size = 1.0
    post_nms_topn = 1
    name = "proposal_5"

    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': nms_threshold,
        'pre_nms_topn': pre_nms_topn,
        'min_size': min_size,
        'post_nms_topn': post_nms_topn,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    scores = np.random.rand(1, 4, 4, 3).astype(np.float32)
    bbox_deltas = np.random.rand(1, 4, 4, 12).astype(np.float32)
    image_info = np.array([[224, 224, 1.0, 224, 224]], dtype=np.float32)
    anchors = np.random.rand(1, 3, 4).astype(np.float32)
    nms_threshold = 0.6
    pre_nms_topn = 4000
    min_size = 10.0
    post_nms_topn = 200
    name = "proposal_6"

    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': nms_threshold,
        'pre_nms_topn': pre_nms_topn,
        'min_size': min_size,
        'post_nms_topn': post_nms_topn,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    scores = np.random.rand(2, 8, 8, 4).astype(np.float32)
    bbox_deltas = np.random.rand(2, 8, 8, 16).astype(np.float32)
    image_info = np.array([[448, 448, 0.8, 448, 448],[448, 448, 0.8, 448, 448]], dtype=np.float32)
    anchors = np.random.rand(1, 4, 4).astype(np.float32)
    nms_threshold = 0.8
    pre_nms_topn = 7000
    min_size = 20.0
    post_nms_topn = 350
    name = "proposal_7"

    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': nms_threshold,
        'pre_nms_topn': pre_nms_topn,
        'min_size': min_size,
        'post_nms_topn': post_nms_topn,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    scores = np.random.rand(1, 16, 16, 2).astype(np.float32)
    bbox_deltas = np.random.rand(1, 16, 16, 8).astype(np.float32)
    image_info = np.array([[896, 896, 1.2, 896, 896]], dtype=np.float32)
    anchors = np.random.rand(1, 2, 4).astype(np.float32)
    nms_threshold = 0.4
    pre_nms_topn = 2000
    min_size = 12.0
    post_nms_topn = 100
    name = "proposal_8"

    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': nms_threshold,
        'pre_nms_topn': pre_nms_topn,
        'min_size': min_size,
        'post_nms_topn': post_nms_topn,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    scores = np.random.rand(3, 2, 2, 1).astype(np.float32)
    bbox_deltas = np.random.rand(3, 2, 2, 4).astype(np.float32)
    image_info = np.array([[112, 112, 0.4, 112, 112],[112, 112, 0.4, 112, 112],[112, 112, 0.4, 112, 112]], dtype=np.float32)
    anchors = np.random.rand(1, 1, 4).astype(np.float32)
    nms_threshold = 0.2
    pre_nms_topn = 500
    min_size = 6.0
    post_nms_topn = 25
    name = "proposal_9"

    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': nms_threshold,
        'pre_nms_topn': pre_nms_topn,
        'min_size': min_size,
        'post_nms_topn': post_nms_topn,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    scores = np.random.rand(1, 32, 32, 6).astype(np.float32)
    bbox_deltas = np.random.rand(1, 32, 32, 24).astype(np.float32)
    image_info = np.array([[1792, 1792, 2.5, 1792, 1792]], dtype=np.float32)
    anchors = np.random.rand(1, 6, 4).astype(np.float32)
    nms_threshold = 0.95
    pre_nms_topn = 9000
    min_size = 40.0
    post_nms_topn = 450
    name = "proposal_10"

    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': nms_threshold,
        'pre_nms_topn': pre_nms_topn,
        'min_size': min_size,
        'post_nms_topn': post_nms_topn,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
