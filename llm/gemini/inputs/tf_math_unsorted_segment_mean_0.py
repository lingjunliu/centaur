
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_unsorted_segment_mean_inputs():
    list_of_inputs = []
    
    # 1. Simple 1D float32
    list_of_inputs.append({
        'data': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'segment_ids': np.array([0, 1, 0, 1], dtype=np.int32),
        'num_segments': 2,
        'name': 'simple_1d'
    })
    
    # 2. 2D data with 1D segment_ids
    list_of_inputs.append({
        'data': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        'segment_ids': np.array([0, 1, 0], dtype=np.int32),
        'num_segments': 2,
        'name': '2d_data_1d_seg'
    })
    
    # 3. Negative segment ids (ignored)
    list_of_inputs.append({
        'data': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        'segment_ids': np.array([-1, 0, -1, 1], dtype=np.int32),
        'num_segments': 2,
        'name': 'neg_seg_ids'
    })
    
    # 4. float16 dtype
    list_of_inputs.append({
        'data': np.array([1.0, 3.0], dtype=np.float16),
        'segment_ids': np.array([0, 0], dtype=np.int32),
        'num_segments': 1,
        'name': 'float16_data'
    })
    
    # 5. Higher dimensional data with multi-dimensional segment_ids
    list_of_inputs.append({
        'data': np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        'segment_ids': np.array([[0, 1], [2, 0]], dtype=np.int32),
        'num_segments': 3,
        'name': 'high_dim'
    })
    
    # 6. Unused segments (output will have 0s)
    list_of_inputs.append({
        'data': np.array([10.0, 20.0], dtype=np.float32),
        'segment_ids': np.array([0, 0], dtype=np.int32),
        'num_segments': 3,
        'name': 'unused_segments'
    })
    
    # 7. All negative segment ids
    list_of_inputs.append({
        'data': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'segment_ids': np.array([-1, -1, -2], dtype=np.int32),
        'num_segments': 2,
        'name': 'all_neg_seg_ids'
    })
    
    # 8. 3D data, 1D segment_ids
    list_of_inputs.append({
        'data': np.ones((3, 2, 2), dtype=np.float32),
        'segment_ids': np.array([0, 1, 1], dtype=np.int32),
        'num_segments': 2,
        'name': '3d_data_1d_seg'
    })
    
    # 9. 4D data, 3D segment_ids
    list_of_inputs.append({
        'data': np.ones((2, 2, 2, 2), dtype=np.float64),
        'segment_ids': np.array([[[0, 1], [1, 0]], [[0, 0], [1, 1]]], dtype=np.int32),
        'num_segments': 2,
        'name': '4d_data_3d_seg'
    })
    
    # 10. Single element data, large num_segments
    list_of_inputs.append({
        'data': np.array([1.5], dtype=np.float32),
        'segment_ids': np.array([4], dtype=np.int32),
        'num_segments': 5,
        'name': 'single_elem'
    })
    
    return list_of_inputs

generated_inputs["tf.math.unsorted_segment_mean"] = tf_math_unsorted_segment_mean_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.unsorted_segment_mean' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.unsorted_segment_mean'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.unsorted_segment_mean', generated_inputs['tf.math.unsorted_segment_mean'], lib="tf", suffix=0)
