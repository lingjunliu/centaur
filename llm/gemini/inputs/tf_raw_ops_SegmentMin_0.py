
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SegmentMin_inputs():
    list_of_inputs = []
    
    # Input 1: 1D float32 data, int32 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_1',
        'data': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        'segment_ids': np.array([0, 0, 1, 1, 2], dtype=np.int32)
    })
    
    # Input 2: 2D float32 data, int32 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_2',
        'data': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        'segment_ids': np.array([0, 0, 1], dtype=np.int32)
    })
    
    # Input 3: 3D float32 data, int64 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_3',
        'data': np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32),
        'segment_ids': np.array([0, 1], dtype=np.int64)
    })
    
    # Input 4: 1D float64 data, int32 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_4',
        'data': np.array([-1.5, -2.5, -3.5], dtype=np.float64),
        'segment_ids': np.array([0, 0, 1], dtype=np.int32)
    })
    
    # Input 5: 2D float64 data, int64 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_5',
        'data': np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64),
        'segment_ids': np.array([0, 1], dtype=np.int64)
    })
    
    # Input 6: 3D float64 data, int32 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_6',
        'data': np.ones((2, 2, 2), dtype=np.float64),
        'segment_ids': np.array([0, 1], dtype=np.int32)
    })
    
    # Input 7: 1D int32 data, int32 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_7',
        'data': np.array([10, 20, 30], dtype=np.int32),
        'segment_ids': np.array([0, 1, 1], dtype=np.int32)
    })
    
    # Input 8: 2D int32 data, int64 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_8',
        'data': np.array([[5, 6], [1, 2], [7, 8]], dtype=np.int32),
        'segment_ids': np.array([0, 0, 1], dtype=np.int64)
    })
    
    # Input 9: 1D int64 data, int64 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_9',
        'data': np.array([-10, -20, -30], dtype=np.int64),
        'segment_ids': np.array([0, 1, 1], dtype=np.int64)
    })
    
    # Input 10: 2D int64 data, int32 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_10',
        'data': np.array([[100, 200], [300, 400]], dtype=np.int64),
        'segment_ids': np.array([0, 1], dtype=np.int32)
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.SegmentMin"] = tf_raw_ops_SegmentMin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SegmentMin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SegmentMin'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SegmentMin', generated_inputs['tf.raw_ops.SegmentMin'], lib="tf", suffix=0)
