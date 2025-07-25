
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_tf_image_generate_bounding_box_proposals_inputs():
    """
    Generates a list of valid inputs for tf.image.generate_bounding_box_proposals.
    The previous attempts failed with `OutOfRangeError: Box dimensions need to be 4`.
    This error, especially on GPU, can be caused by an empty list of proposals after
    the filtering stages (clipping, min_size). This version creates robust inputs
    that are guaranteed to produce valid proposals to avoid this potential issue.
    """
    list_of_inputs = []

    def _create_robust_input(
        num_images, height, width, num_anchors,
        nms_threshold=0.7, pre_nms_topn=2000, min_size=1.0, post_nms_topn=100,
        name=None):

        scores = np.random.uniform(0.6, 1.0, (num_images, height, width, num_anchors)).astype(np.float32)
        bbox_deltas = (np.random.rand(num_images, height, width, 4 * num_anchors) * 0.1 - 0.05).astype(np.float32)

        img_h, img_w = 600, 800
        image_info = np.array([[img_h, img_w, 1.0, img_h, img_w]] * num_images, dtype=np.float32)

        anchors_list = []
        for _ in range(num_anchors):
            y1 = np.random.uniform(50, 200)
            x1 = np.random.uniform(50, 300)
            y2 = y1 + np.random.uniform(50, 150)
            x2 = x1 + np.random.uniform(50, 150)
            anchors_list.append([y1, x1, y2, x2])
        anchors_2d = np.array(anchors_list, dtype=np.float32)
        anchors = np.expand_dims(anchors_2d, 0)

        total_proposals = height * width * num_anchors
        if pre_nms_topn > total_proposals:
            pre_nms_topn = total_proposals
        
        if post_nms_topn > pre_nms_topn:
            post_nms_topn = pre_nms_topn

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

    _create_robust_input(
        num_images=1, height=32, width=32, num_anchors=9, name="robust_case_1"
    )

    _create_robust_input(
        num_images=2, height=16, width=16, num_anchors=5, name="robust_multi_image"
    )

    _create_robust_input(
        num_images=1, height=20, width=20, num_anchors=4, nms_threshold=0.95, name="robust_high_nms"
    )

    _create_robust_input(
        num_images=1, height=20, width=20, num_anchors=12, nms_threshold=0.2, pre_nms_topn=500, post_nms_topn=50, name="robust_low_nms"
    )

    _create_robust_input(
        num_images=1, height=4, width=4, num_anchors=2, pre_nms_topn=30, post_nms_topn=50, name="robust_fewer_proposals"
    )

    _create_robust_input(
        num_images=1, height=15, width=15, num_anchors=3, min_size=40.0, name="robust_large_min_size"
    )

    _create_robust_input(
        num_images=1, height=10, width=10, num_anchors=3, name="robust_zero_deltas"
    )
    list_of_inputs[-1]['bbox_deltas'] = np.zeros_like(list_of_inputs[-1]['bbox_deltas'])

    _create_robust_input(
        num_images=1, height=8, width=8, num_anchors=9, name="robust_small_feature_map"
    )

    _create_robust_input(
        num_images=1, height=16, width=16, num_anchors=15, name="robust_more_anchors"
    )

    _create_robust_input(
        num_images=2, height=20, width=20, num_anchors=6, pre_nms_topn=500, post_nms_topn=50, name="robust_low_topn"
    )

    return list_of_inputs

generated_inputs["tf.image.generate_bounding_box_proposals"] = generate_tf_image_generate_bounding_box_proposals_inputs()

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
