
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_image_generate_bounding_box_proposals_inputs():
    list_of_inputs = []

    # The TensorFlow runtime throws an error `anchors must be rank 3`, which
    # contradicts the public documentation (that states rank 2). When a rank-3
    # tensor is provided, a secondary error `OutOfRangeError: Box dimensions need
    # to be 4` occurs if the first dimension of `anchors` matches the batch size
    # of other inputs. This suggests the C++ kernel expects a broadcastable
    # anchor tensor of shape `[1, num_anchors, 4]`, which is then applied to
    # every image in the batch. This implementation adheres to that finding.
    def _generate_tensors(num_images, height, width, num_anchors, dtype=np.float32):
        scores = np.random.rand(num_images, height, width, num_anchors).astype(dtype)
        bbox_deltas = np.random.randn(num_images, height, width, 4 * num_anchors).astype(dtype)
        image_info = np.array([[height * 16, width * 16, 1.0, 0.0, 0.0]] * num_images, dtype=dtype)
        
        y1 = np.random.randint(0, 50, size=(num_anchors, 1))
        x1 = np.random.randint(0, 50, size=(num_anchors, 1))
        h = np.random.randint(16, 100, size=(num_anchors, 1))
        w = np.random.randint(16, 100, size=(num_anchors, 1))
        y2 = y1 + h
        x2 = x1 + w
        anchors_2d = np.hstack([y1, x1, y2, x2]).astype(dtype)

        # Per runtime errors, anchors must be rank 3.
        # It must be broadcastable across the batch, so shape is [1, num_anchors, 4].
        anchors_3d = np.expand_dims(anchors_2d, 0)
        
        return scores, bbox_deltas, image_info, anchors_3d

    # Input 1: Basic case (num_images=1)
    scores, bbox_deltas, image_info, anchors = _generate_tensors(1, 10, 10, 9)
    input_dict = {
        'scores': scores, 'bbox_deltas': bbox_deltas, 'image_info': image_info,
        'anchors': anchors, 'nms_threshold': 0.7, 'pre_nms_topn': 6000,
        'min_size': 16.0, 'post_nms_topn': 300, 'name': 'basic_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of images (num_images=2)
    scores, bbox_deltas, image_info, anchors = _generate_tensors(2, 12, 12, 5)
    input_dict = {
        'scores': scores, 'bbox_deltas': bbox_deltas, 'image_info': image_info,
        'anchors': anchors, 'nms_threshold': 0.7, 'pre_nms_topn': 6000,
        'min_size': 16.0, 'post_nms_topn': 300, 'name': 'batch_of_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Lower nms_threshold
    scores, bbox_deltas, image_info, anchors = _generate_tensors(1, 10, 10, 9)
    input_dict = {
        'scores': scores, 'bbox_deltas': bbox_deltas, 'image_info': image_info,
        'anchors': anchors, 'nms_threshold': 0.5, 'pre_nms_topn': 6000,
        'min_size': 16.0, 'post_nms_topn': 300, 'name': 'low_nms'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Smaller pre_nms_topn
    scores, bbox_deltas, image_info, anchors = _generate_tensors(1, 20, 20, 10)
    input_dict = {
        'scores': scores, 'bbox_deltas': bbox_deltas, 'image_info': image_info,
        'anchors': anchors, 'nms_threshold': 0.7, 'pre_nms_topn': 2000,
        'min_size': 16.0, 'post_nms_topn': 300, 'name': 'small_pre_nms'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Smaller post_nms_topn
    scores, bbox_deltas, image_info, anchors = _generate_tensors(1, 20, 20, 10)
    input_dict = {
        'scores': scores, 'bbox_deltas': bbox_deltas, 'image_info': image_info,
        'anchors': anchors, 'nms_threshold': 0.7, 'pre_nms_topn': 6000,
        'min_size': 16.0, 'post_nms_topn': 100, 'name': 'small_post_nms'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger min_size
    scores, bbox_deltas, image_info, anchors = _generate_tensors(1, 15, 15, 9)
    input_dict = {
        'scores': scores, 'bbox_deltas': bbox_deltas, 'image_info': image_info,
        'anchors': anchors, 'nms_threshold': 0.7, 'pre_nms_topn': 6000,
        'min_size': 32.0, 'post_nms_topn': 300, 'name': 'large_min_size'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single anchor case
    scores, bbox_deltas, image_info, anchors = _generate_tensors(1, 8, 8, 1)
    input_dict = {
        'scores': scores, 'bbox_deltas': bbox_deltas, 'image_info': image_info,
        'anchors': anchors, 'nms_threshold': 0.8, 'pre_nms_topn': 5000,
        'min_size': 10.0, 'post_nms_topn': 200, 'name': 'single_anchor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All scores are identical
    scores, bbox_deltas, image_info, anchors = _generate_tensors(1, 5, 5, 3)
    scores = np.full_like(scores, 0.9, dtype=scores.dtype)
    input_dict = {
        'scores': scores, 'bbox_deltas': bbox_deltas, 'image_info': image_info,
        'anchors': anchors, 'nms_threshold': 0.7, 'pre_nms_topn': 6000,
        'min_size': 16.0, 'post_nms_topn': 300, 'name': 'identical_scores'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Negative and positive scores
    scores, bbox_deltas, image_info, anchors = _generate_tensors(1, 10, 10, 9)
    scores = (scores - 0.5) * 2
    input_dict = {
        'scores': scores, 'bbox_deltas': bbox_deltas, 'image_info': image_info,
        'anchors': anchors, 'nms_threshold': 0.7, 'pre_nms_topn': 6000,
        'min_size': 16.0, 'post_nms_topn': 300, 'name': 'negative_scores'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Zero bbox_deltas
    scores, bbox_deltas, image_info, anchors = _generate_tensors(1, 10, 10, 9)
    bbox_deltas = np.zeros_like(bbox_deltas)
    input_dict = {
        'scores': scores, 'bbox_deltas': bbox_deltas, 'image_info': image_info,
        'anchors': anchors, 'nms_threshold': 0.7, 'pre_nms_topn': 6000,
        'min_size': 16.0, 'post_nms_topn': 300, 'name': 'zero_deltas'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

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

check_valid('tf.image.generate_bounding_box_proposals', generated_inputs['tf.image.generate_bounding_box_proposals'], lib="tf", suffix=0)
