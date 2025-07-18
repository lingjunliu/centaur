
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# Use a named function to avoid potential lambda issues with deepcopy/serialization
def _get_length_func(elem):
  """Returns the length of a tensor element."""
  return tf.shape(elem)[0]

def bucket_by_sequence_length_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.bucket_by_sequence_length.
    """
    list_of_inputs = []

    # The test harness expects the input data for the dataset under a special key.
    # The error "input does not have inner values" suggests this key is missing.
    # We will use the key 'dataset' to provide the data that the transformation function will be applied to.

    # Case 1: Basic case with default padding.
    case1_data = [np.arange(i, dtype=np.int32) for i in [2, 3, 4, 1, 5, 6, 7, 8, 10, 11, 12, 15]]
    input_1 = {
        'dataset': case1_data,
        'element_length_func': [_get_length_func],
        'bucket_boundaries': [5, 10],
        'bucket_batch_sizes': [4, 4, 4],
        'padded_shapes': (None,),
        'padding_values': np.array(0, dtype=np.int32),
        'pad_to_bucket_boundary': False,
        'no_padding': False,
        'drop_remainder': False,
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    # Case 2: Custom negative integer padding value.
    case2_data = [np.arange(i, dtype=np.int32) for i in range(1, 9)] + [np.arange(i, dtype=np.int32) for i in range(10, 18)] + [np.arange(i, dtype=np.int32) for i in range(20, 28)]
    input_2 = {
        'dataset': case2_data,
        'element_length_func': [_get_length_func],
        'bucket_boundaries': [10, 20],
        'bucket_batch_sizes': [8, 8, 8],
        'padded_shapes': (None,),
        'padding_values': np.array(-1, dtype=np.int32),
        'pad_to_bucket_boundary': False,
        'no_padding': False,
        'drop_remainder': False,
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    # Case 3: Pad to bucket boundary enabled.
    case3_data = [np.arange(i, dtype=np.int32) for i in range(1, 8)] * 3
    input_3 = {
        'dataset': case3_data,
        'element_length_func': [_get_length_func],
        'bucket_boundaries': [8, 16],
        'bucket_batch_sizes': [10, 10, 10],
        'padded_shapes': (None,),
        'padding_values': np.array(0, dtype=np.int32),
        'pad_to_bucket_boundary': True,
        'no_padding': False,
        'drop_remainder': False,
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    # Case 4: Pad to bucket boundary with drop_remainder, from docs.
    case4_data = [
      np.array([0], dtype=np.int32), np.array([1, 2, 3, 4], dtype=np.int32), np.array([5, 6, 7], dtype=np.int32),
      np.array([7, 8, 9, 10, 11], dtype=np.int32), np.array([13, 14, 15, 16, 19, 20], dtype=np.int32), np.array([21, 22], dtype=np.int32)
    ]
    input_4 = {
        'dataset': case4_data,
        'element_length_func': [_get_length_func],
        'bucket_boundaries': [4, 7],
        'bucket_batch_sizes': [2, 2, 2],
        'padded_shapes': (None,),
        'padding_values': np.array(-1, dtype=np.int32),
        'pad_to_bucket_boundary': True,
        'no_padding': False,
        'drop_remainder': True,
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    # Case 5: More buckets and varying batch sizes.
    case5_data = ([np.arange(4, dtype=np.int64)] * 32 +
               [np.arange(6, dtype=np.int64)] * 16 +
               [np.arange(12, dtype=np.int64)] * 8 +
               [np.arange(18, dtype=np.int64)] * 4 +
               [np.arange(22, dtype=np.int64)] * 2)
    input_5 = {
        'dataset': case5_data,
        'element_length_func': [_get_length_func],
        'bucket_boundaries': [5, 10, 15, 20],
        'bucket_batch_sizes': [32, 16, 8, 4, 2],
        'padded_shapes': (None,),
        'padding_values': np.array(0, dtype=np.int64),
        'pad_to_bucket_boundary': False,
        'no_padding': False,
        'drop_remainder': False,
    }
    list_of_inputs.append(copy.deepcopy(input_5))

    # Case 6: Explicitly set padded_shapes.
    case6_data = [np.arange(i, dtype=np.int32) for i in range(10, 20)] + [np.arange(i, dtype=np.int32) for i in range(50, 55)]
    input_6 = {
        'dataset': case6_data,
        'element_length_func': [_get_length_func],
        'bucket_boundaries': [50],
        'bucket_batch_sizes': [10, 5],
        'padded_shapes': (60,),
        'padding_values': np.array(99, dtype=np.int32),
        'pad_to_bucket_boundary': False,
        'no_padding': False,
        'drop_remainder': True,
    }
    list_of_inputs.append(copy.deepcopy(input_6))

    # Case 7: Single bucket boundary and float padding value.
    case7_data = [np.arange(i, dtype=np.float32) for i in range(1, 9)] + [np.arange(i, dtype=np.float32) for i in range(32, 36)]
    input_7 = {
        'dataset': case7_data,
        'element_length_func': [_get_length_func],
        'bucket_boundaries': [32],
        'bucket_batch_sizes': [8, 4],
        'padded_shapes': (None,),
        'padding_values': np.array(0.0, dtype=np.float32),
        'pad_to_bucket_boundary': False,
        'no_padding': False,
        'drop_remainder': False,
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    return list_of_inputs

generated_inputs["tf.data.experimental.bucket_by_sequence_length"] = bucket_by_sequence_length_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.bucket_by_sequence_length' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.bucket_by_sequence_length'.")

check_valid('tf.data.experimental.bucket_by_sequence_length', generated_inputs['tf.data.experimental.bucket_by_sequence_length'], lib="tf", suffix=0)
