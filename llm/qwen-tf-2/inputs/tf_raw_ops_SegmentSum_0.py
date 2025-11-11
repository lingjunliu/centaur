
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def generate_segment_sum_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with float data
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    
    input_dict = {
        "name": "SegmentSum_1",
        "data": data,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Integer data with negative values
    data = np.array([[1, -2], [3, -4], [5, -6]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    
    input_dict = {
        "name": "SegmentSum_2",
        "data": data,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Float data with multiple segments (sorted)
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    
    input_dict = {
        "name": "SegmentSum_3",
        "data": data,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Complex data with repeated segments
    data = np.array([[1+2j, 3+4j], [5+6j, 7+8j], [9+10j, 11+12j]], dtype=np.complex64)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    
    input_dict = {
        "name": "SegmentSum_4",
        "data": data,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single dimension data with repeated segment IDs (sorted)
    data = np.array([1, 2, 3, 4], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    
    input_dict = {
        "name": "SegmentSum_5",
        "data": data,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Float data with empty segments (sorted)
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    
    input_dict = {
        "name": "SegmentSum_6",
        "data": data,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Large number of segments with varying values (sorted)
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    
    input_dict = {
        "name": "SegmentSum_7",
        "data": data,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Large data with multiple segments and negative values (sorted)
    data = np.array([[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]], dtype=np.float32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    
    input_dict = {
        "name": "SegmentSum_8",
        "data": data,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Different data types with repeated segment IDs (sorted)
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    segment_ids = np.array([0, 0, 1], dtype=np.int64)
    
    input_dict = {
        "name": "SegmentSum_9",
        "data": data,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Int64 data with different segment IDs (sorted)
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64)
    segment_ids = np.array([0, 1, 0, 1], dtype=np.int64)
    
    input_dict = {
        "name": "SegmentSum_10",
        "data": data,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.SegmentSum"] = generate_segment_sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SegmentSum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SegmentSum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SegmentSum', generated_inputs['tf.raw_ops.SegmentSum'], lib="tf", suffix=0)
