
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def get_generate_bounding_box_proposals_inputs():
    list_of_inputs = []

    # The error "Box dimensions need to be 4" and an inspection of the op's
    # shape function in TensorFlow's source code reveals that the `image_info`
    # tensor must have a shape of `[num_images, 4]`, contrary to the
    # documentation which states `[num_images, 5]`. The 4 values typically
    # represent `[image_height, image_width, y_scale, x_scale]`.
    #
    # The previous error "'anchors' must be rank 3 but is rank 2" was also
    # likely a side-effect of the incorrect `image_info` shape, as the op's
    # shape function expects a rank-2 tensor for `anchors`. We revert `anchors`
    # to be rank-2.

    def _generate_anchors(num_anchors, max_coord=100.0, min_size=10.0, dtype=np.float32):
        anchors = []
        for _ in range(num_anchors):
            y1 = np.random.uniform(0, max_coord - min_size)
            x1 = np.random.uniform(0, max_coord - min_size)
            y2 = np.random.uniform(y1 + min_size, max_coord)
            x2 = np.random.uniform(x1 + min_size, max_coord)
            anchors.append([y1, x1, y2, x2])
        # Revert anchors to rank 2 as per the op's shape function
        return np.array(anchors, dtype=dtype)

    # Input 1: Basic case with fixes applied
    num_images, height, width, num_anchors = 1, 10, 10, 9
    scores = np.random.rand(num_images, height, width, num_anchors).astype(np.float32)
    bbox_deltas = (np.random.rand(num_images, height, width, 4 * num_anchors) - 0.5).astype(np.float32)
    image_info = np.array([[400.0, 600.0, 1.0, 1.0]] * num_images, dtype=np.float32)
    anchors = _generate_anchors(num_anchors)
    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': 0.7,
        'pre_nms_topn': 6000,
        'min_size': 16.0,
        'post_nms_topn': 300,
        'name': 'basic_case_fixed'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple images and different parameters
    num_images, height, width, num_anchors = 2, 8, 12, 5
    scores = np.random.rand(num_images, height, width, num_anchors).astype(np.float32)
    bbox_deltas = (np.random.rand(num_images, height, width, 4 * num_anchors) * 2 - 1).astype(np.float32)
    image_info = np.array([[320.0, 480.0, 1.0, 1.0], [240.0, 320.0, 1.2, 1.2]], dtype=np.float32)
    anchors = _generate_anchors(num_anchors)
    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': 0.5,
        'pre_nms_topn': 1000,
        'min_size': 10.0,
        'post_nms_topn': 100,
        'name': 'multi_image_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Minimal dimensions and zero deltas
    num_images, height, width, num_anchors = 1, 1, 1, 1
    scores = np.random.rand(num_images, height, width, num_anchors).astype(np.float32)
    bbox_deltas = np.zeros((num_images, height, width, 4 * num_anchors), dtype=np.float32)
    image_info = np.array([[100.0, 100.0, 1.0, 1.0]] * num_images, dtype=np.float32)
    anchors = _generate_anchors(num_anchors, max_coord=50)
    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': 0.9,
        'pre_nms_topn': 10,
        'min_size': 1.0,
        'post_nms_topn': 5,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large pre_nms_topn
    num_images, height, width, num_anchors = 1, 5, 5, 4
    scores = np.random.rand(num_images, height, width, num_anchors).astype(np.float32)
    bbox_deltas = (np.random.rand(num_images, height, width, 4 * num_anchors) - 0.5).astype(np.float32)
    image_info = np.array([[200.0, 200.0, 1.0, 1.0]] * num_images, dtype=np.float32)
    anchors = _generate_anchors(num_anchors, max_coord=80)
    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': 0.7,
        'pre_nms_topn': 500,
        'min_size': 16.0,
        'post_nms_topn': 200,
        'name': 'large_topn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small pre_nms_topn and post_nms_topn
    num_images, height, width, num_anchors = 1, 20, 20, 10
    scores = np.random.rand(num_images, height, width, num_anchors).astype(np.float32)
    bbox_deltas = (np.random.rand(num_images, height, width, 4 * num_anchors) - 0.5).astype(np.float32)
    image_info = np.array([[800.0, 800.0, 1.0, 1.0]] * num_images, dtype=np.float32)
    anchors = _generate_anchors(num_anchors, max_coord=200)
    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': 0.6,
        'pre_nms_topn': 100,
        'min_size': 20.0,
        'post_nms_topn': 50,
        'name': 'small_topn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Very low NMS threshold
    num_images, height, width, num_anchors = 1, 15, 15, 8
    scores = np.random.rand(num_images, height, width, num_anchors).astype(np.float32)
    bbox_deltas = (np.random.rand(num_images, height, width, 4 * num_anchors) - 0.5).astype(np.float32)
    image_info = np.array([[600.0, 600.0, 1.5, 1.5]] * num_images, dtype=np.float32)
    anchors = _generate_anchors(num_anchors, max_coord=150)
    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': 0.1,
        'pre_nms_topn': 2000,
        'min_size': 16.0,
        'post_nms_topn': 300,
        'name': 'low_nms'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large min_size
    num_images, height, width, num_anchors = 1, 10, 10, 9
    scores = np.random.rand(num_images, height, width, num_anchors).astype(np.float32)
    bbox_deltas = (np.random.rand(num_images, height, width, 4 * num_anchors) - 0.5).astype(np.float32)
    image_info = np.array([[400.0, 600.0, 1.0, 1.0]] * num_images, dtype=np.float32)
    anchors = _generate_anchors(num_anchors)
    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': 0.7,
        'pre_nms_topn': 6000,
        'min_size': 100.0,
        'post_nms_topn': 300,
        'name': 'large_min_size'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 dtype
    num_images, height, width, num_anchors = 1, 10, 10, 9
    scores = np.random.rand(num_images, height, width, num_anchors).astype(np.float64)
    bbox_deltas = (np.random.rand(num_images, height, width, 4 * num_anchors) - 0.5).astype(np.float64)
    image_info = np.array([[400.0, 600.0, 1.0, 1.0]] * num_images, dtype=np.float64)
    anchors = _generate_anchors(num_anchors, dtype=np.float64)
    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': 0.7,
        'pre_nms_topn': 6000,
        'min_size': 16.0,
        'post_nms_topn': 300,
        'name': 'float64_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Extreme scores (all ones)
    num_images, height, width, num_anchors = 1, 6, 6, 3
    scores = np.ones((num_images, height, width, num_anchors), dtype=np.float32)
    bbox_deltas = (np.random.rand(num_images, height, width, 4 * num_anchors) - 0.5).astype(np.float32)
    image_info = np.array([[180.0, 180.0, 1.0, 1.0]] * num_images, dtype=np.float32)
    anchors = _generate_anchors(num_anchors, max_coord=50)
    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': 0.5,
        'pre_nms_topn': 100,
        'min_size': 10.0,
        'post_nms_topn': 50,
        'name': 'ones_scores'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative scores and min_size=0
    num_images, height, width, num_anchors = 1, 4, 4, 2
    scores = (np.random.rand(num_images, height, width, num_anchors) - 0.5).astype(np.float32)
    bbox_deltas = (np.random.rand(num_images, height, width, 4 * num_anchors) - 0.5).astype(np.float32)
    image_info = np.array([[128.0, 128.0, 1.0, 1.0]] * num_images, dtype=np.float32)
    anchors = _generate_anchors(num_anchors, max_coord=32)
    input_dict = {
        'scores': scores,
        'bbox_deltas': bbox_deltas,
        'image_info': image_info,
        'anchors': anchors,
        'nms_threshold': 0.8,
        'pre_nms_topn': 32,
        'min_size': 0.0,
        'post_nms_topn': 10,
        'name': 'negative_scores'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.generate_bounding_box_proposals"] = get_generate_bounding_box_proposals_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.generate_bounding_box_proposals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.generate_bounding_box_proposals'.")

check_valid('tf.image.generate_bounding_box_proposals', generated_inputs['tf.image.generate_bounding_box_proposals'], lib="tf", suffix=0)
