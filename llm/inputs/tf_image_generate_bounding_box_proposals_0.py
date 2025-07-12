
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np

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
    name = "test_op"

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

    # Input 2: Different shapes and values
    scores = np.random.rand(2, 5, 5, 5).astype(np.float32)
    bbox_deltas = np.random.rand(2, 5, 5, 20).astype(np.float32)
    image_info = np.array([[300, 400, 0.5, 300, 400], [300, 400, 0.5, 300, 400]], dtype=np.float32)
    anchors = np.random.rand(2, 5, 4).astype(np.float32)
    nms_threshold = 0.5
    pre_nms_topn = 3000
    min_size = 8.0
    post_nms_topn = 150
    name = "test_op2"
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

     # Input 3: Smaller values and shapes
    scores = np.random.rand(1, 3, 3, 2).astype(np.float32)
    bbox_deltas = np.random.rand(1, 3, 3, 8).astype(np.float32)
    image_info = np.array([[100, 100, 1.0, 100, 100]], dtype=np.float32)
    anchors = np.random.rand(1, 2, 4).astype(np.float32)
    nms_threshold = 0.9
    pre_nms_topn = 1000
    min_size = 4.0
    post_nms_topn = 50
    name = "test_op3"
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

    # Input 4: Different image info and anchor values
    scores = np.random.rand(1, 8, 8, 4).astype(np.float32)
    bbox_deltas = np.random.rand(1, 8, 8, 16).astype(np.float32)
    image_info = np.array([[800, 600, 0.8, 800, 600]], dtype=np.float32)
    anchors = np.random.rand(1, 4, 4).astype(np.float32)
    nms_threshold = 0.6
    pre_nms_topn = 4000
    min_size = 12.0
    post_nms_topn = 200
    name = "test_op4"
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

    # Input 5: Different num_images
    scores = np.random.rand(3, 4, 4, 3).astype(np.float32)
    bbox_deltas = np.random.rand(3, 4, 4, 12).astype(np.float32)
    image_info = np.array([[200, 300, 0.7, 200, 300],[200, 300, 0.7, 200, 300],[200, 300, 0.7, 200, 300]], dtype=np.float32)
    anchors = np.random.rand(1, 3, 4).astype(np.float32)
    nms_threshold = 0.8
    pre_nms_topn = 5000
    min_size = 10.0
    post_nms_topn = 250
    name = "test_op5"
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

    # Input 6: Different aspect ratio
    scores = np.random.rand(1, 6, 12, 2).astype(np.float32)
    bbox_deltas = np.random.rand(1, 6, 12, 8).astype(np.float32)
    image_info = np.array([[400, 200, 1.2, 400, 200]], dtype=np.float32)
    anchors = np.random.rand(1, 2, 4).astype(np.float32)
    nms_threshold = 0.4
    pre_nms_topn = 2000
    min_size = 6.0
    post_nms_topn = 100
    name = "test_op6"
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

    # Input 7: Identity scale
    scores = np.random.rand(1, 7, 7, 3).astype(np.float32)
    bbox_deltas = np.random.rand(1, 7, 7, 12).astype(np.float32)
    image_info = np.array([[500, 500, 1.0, 500, 500]], dtype=np.float32)
    anchors = np.random.rand(1, 3, 4).astype(np.float32)
    nms_threshold = 0.75
    pre_nms_topn = 6000
    min_size = 16.0
    post_nms_topn = 300
    name = "test_op7"
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
    scores = np.random.rand(1, 10, 10, 3).astype(np.float32)
    bbox_deltas = np.random.rand(1, 10, 10, 12).astype(np.float32)
    image_info = np.array([[1024, 768, 1.0, 1024, 768]], dtype=np.float32)
    anchors = np.random.rand(1, 3, 4).astype(np.float32)
    nms_threshold = 0.7
    pre_nms_topn = 6000
    min_size = 16.0
    post_nms_topn = 300
    name = "test_op8"

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

     # Input 9 - Lower resolution inputs
    scores = np.random.rand(1, 5, 5, 2).astype(np.float32)
    bbox_deltas = np.random.rand(1, 5, 5, 8).astype(np.float32)
    image_info = np.array([[256, 256, 1.0, 256, 256]], dtype=np.float32)
    anchors = np.random.rand(1, 2, 4).astype(np.float32)
    nms_threshold = 0.6
    pre_nms_topn = 3000
    min_size = 8.0
    post_nms_topn = 150
    name = "test_op9"

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

    # Input 10 - Larger pre/post nms
    scores = np.random.rand(1, 8, 8, 4).astype(np.float32)
    bbox_deltas = np.random.rand(1, 8, 8, 16).astype(np.float32)
    image_info = np.array([[512, 512, 1.0, 512, 512]], dtype=np.float32)
    anchors = np.random.rand(1, 4, 4).astype(np.float32)
    nms_threshold = 0.5
    pre_nms_topn = 8000
    min_size = 12.0
    post_nms_topn = 400
    name = "test_op10"

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

generated_inputs["tf.image.generate_bounding_box_proposals"] = tf_image_generate_bounding_box_proposals_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.generate_bounding_box_proposals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.generate_bounding_box_proposals'.")

check_valid('tf.image.generate_bounding_box_proposals', generated_inputs['tf.image.generate_bounding_box_proposals'], lib="tf", suffix=0)
