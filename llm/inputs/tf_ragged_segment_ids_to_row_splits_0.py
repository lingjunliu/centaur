
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_segment_ids_to_row_splits_inputs():
    list_of_inputs = []

    # Input 1
    segment_ids = np.array([0, 0, 0, 2, 2, 3, 4, 4, 4], dtype=np.int32)
    num_segments = 6
    out_type = np.int64
    name = "example_1"
    input_dict = {"segment_ids": segment_ids, "num_segments": num_segments, "out_type": np.dtype(out_type), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    segment_ids = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    num_segments = 3
    out_type = np.int32
    name = "example_2"
    input_dict = {"segment_ids": segment_ids, "num_segments": num_segments, "out_type": np.dtype(out_type), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    segment_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    num_segments = 5
    out_type = np.int64
    name = "example_3"
    input_dict = {"segment_ids": segment_ids, "num_segments": num_segments, "out_type": np.dtype(out_type), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    segment_ids = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    num_segments = 1
    out_type = np.int32
    name = "example_4"
    input_dict = {"segment_ids": segment_ids, "num_segments": num_segments, "out_type": np.dtype(out_type), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    segment_ids = np.array([2, 2, 2, 2], dtype=np.int32)
    num_segments = 3
    out_type = np.int64
    name = "example_5"
    input_dict = {"segment_ids": segment_ids, "num_segments": num_segments, "out_type": np.dtype(out_type), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    segment_ids = np.array([0], dtype=np.int64)
    num_segments = 1
    out_type = np.int32
    name = "example_6"
    input_dict = {"segment_ids": segment_ids, "num_segments": num_segments, "out_type": np.dtype(out_type), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    segment_ids = np.array([], dtype=np.int32)
    num_segments = 0
    out_type = np.int64
    name = "example_7"
    input_dict = {"segment_ids": segment_ids, "num_segments": num_segments, "out_type": np.dtype(out_type), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    segment_ids = np.array([5, 5, 5, 6, 6, 7], dtype=np.int64)
    num_segments = 8
    out_type = np.int32
    name = "example_8"
    input_dict = {"segment_ids": segment_ids, "num_segments": num_segments, "out_type": np.dtype(out_type), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    segment_ids = np.array([10, 10, 11, 12], dtype=np.int32)
    num_segments = 13
    out_type = np.int64
    name = "example_9"
    input_dict = {"segment_ids": segment_ids, "num_segments": num_segments, "out_type": np.dtype(out_type), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    segment_ids = np.array([0, 0, 1, 2, 2, 2, 3, 3], dtype=np.int64)
    num_segments = 4
    out_type = np.int32
    name = "example_10"
    input_dict = {"segment_ids": segment_ids, "num_segments": num_segments, "out_type": np.dtype(out_type), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    segment_ids = np.array([0, 0, 1, 2, 2, 2, 3, 3], dtype=np.int32)
    num_segments = None
    out_type = np.int64
    name = "example_11"
    input_dict = {"segment_ids": segment_ids, "num_segments": num_segments, "out_type": np.dtype(out_type), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ragged.segment_ids_to_row_splits"] = tf_ragged_segment_ids_to_row_splits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ragged.segment_ids_to_row_splits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.segment_ids_to_row_splits'.")

check_valid('tf.ragged.segment_ids_to_row_splits', generated_inputs['tf.ragged.segment_ids_to_row_splits'], lib="tf", suffix=0)
