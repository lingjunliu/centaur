
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
    nms_threshold = 0.5
    pre_nms_topn = 100
    min_size = 8.0
    post_nms_topn = 50
    name = "proposal_generator"

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
    list_of_inputs.append(input_dict)

    # Input 2
    scores = np.random.rand(2, 5, 5, 5).astype(np.float32)
    bbox_deltas = np.random.rand(2, 5, 5, 20).astype(np.float32)
    image_info = np.array([[300, 400, 0.5, 300, 400], [300, 400, 0.5, 300, 400]]).astype(np.float32)
    anchors = np.random.rand(1, 5, 4).astype(np.float32)
    nms_threshold = 0.6
    pre_nms_topn = 200
    min_size = 10.0
    post_nms_topn = 100
    name = None

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
    list_of_inputs.append(input_dict)

    # Input 3
    scores = np.random.rand(1, 20, 20, 2).astype(np.float32)
    bbox_deltas = np.random.rand(1, 20, 20, 8).astype(np.float32)
    image_info = np.array([[800, 1000, 1.2, 800, 1000]]).astype(np.float32)
    anchors = np.random.rand(1, 2, 4).astype(np.float32)
    nms_threshold = 0.4
    pre_nms_topn = 300
    min_size = 12.0
    post_nms_topn = 150
    name = "region_proposals"

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
    list_of_inputs.append(input_dict)

    # Input 4
    scores = np.random.rand(1, 8, 8, 4).astype(np.float32)
    bbox_deltas = np.random.rand(1, 8, 8, 16).astype(np.float32)
    image_info = np.array([[400, 500, 0.8, 400, 500]]).astype(np.float32)
    anchors = np.random.rand(1, 4, 4).astype(np.float32)
    nms_threshold = 0.9
    pre_nms_topn = 400
    min_size = 14.0
    post_nms_topn = 200
    name = "roi_generator"

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
    list_of_inputs.append(input_dict)

    # Input 5
    scores = np.random.rand(1, 15, 15, 6).astype(np.float32)
    bbox_deltas = np.random.rand(1, 15, 15, 24).astype(np.float32)
    image_info = np.array([[700, 900, 1.1, 700, 900]]).astype(np.float32)
    anchors = np.random.rand(1, 6, 4).astype(np.float32)
    nms_threshold = 0.3
    pre_nms_topn = 500
    min_size = 15.0
    post_nms_topn = 250
    name = "fast_rcnn_proposals"

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
    list_of_inputs.append(input_dict)

    # Input 6
    scores = np.random.rand(1, 7, 7, 3).astype(np.float32)
    bbox_deltas = np.random.rand(1, 7, 7, 12).astype(np.float32)
    image_info = np.array([[350, 450, 0.6, 350, 450]]).astype(np.float32)
    anchors = np.random.rand(1, 3, 4).astype(np.float32)
    nms_threshold = 0.8
    pre_nms_topn = 1000
    min_size = 18.0
    post_nms_topn = 300
    name = None

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
    list_of_inputs.append(input_dict)

    # Input 7
    scores = np.random.rand(1, 25, 25, 1).astype(np.float32)
    bbox_deltas = np.random.rand(1, 25, 25, 4).astype(np.float32)
    image_info = np.array([[900, 1100, 1.3, 900, 1100]]).astype(np.float32)
    anchors = np.random.rand(1, 1, 4).astype(np.float32)
    nms_threshold = 0.2
    pre_nms_topn = 1500
    min_size = 20.0
    post_nms_topn = 350
    name = "object_proposals"

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
    list_of_inputs.append(input_dict)

    # Input 8
    scores = np.random.rand(1, 6, 6, 2).astype(np.float32)
    bbox_deltas = np.random.rand(1, 6, 6, 8).astype(np.float32)
    image_info = np.array([[300, 350, 0.7, 300, 350]]).astype(np.float32)
    anchors = np.random.rand(1, 2, 4).astype(np.float32)
    nms_threshold = 0.75
    pre_nms_topn = 2000
    min_size = 22.0
    post_nms_topn = 400
    name = "refined_proposals"

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
    list_of_inputs.append(input_dict)

    # Input 9
    scores = np.random.rand(1, 30, 30, 4).astype(np.float32)
    bbox_deltas = np.random.rand(1, 30, 30, 16).astype(np.float32)
    image_info = np.array([[1000, 1200, 1.4, 1000, 1200]]).astype(np.float32)
    anchors = np.random.rand(1, 4, 4).astype(np.float32)
    nms_threshold = 0.1
    pre_nms_topn = 3000
    min_size = 24.0
    post_nms_topn = 450
    name = "candidate_proposals"

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
    list_of_inputs.append(input_dict)

    # Input 10
    scores = np.random.rand(1, 9, 9, 5).astype(np.float32)
    bbox_deltas = np.random.rand(1, 9, 9, 20).astype(np.float32)
    image_info = np.array([[450, 550, 0.9, 450, 550]]).astype(np.float32)
    anchors = np.random.rand(1, 5, 4).astype(np.float32)
    nms_threshold = 0.65
    pre_nms_topn = 4000
    min_size = 26.0
    post_nms_topn = 500
    name = None

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
