
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_segment_mean_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = 2
    name = "segment_mean_1"
    sparse_gradient = False
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "num_segments": num_segments, "name": name, "sparse_gradient": sparse_gradient}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    num_segments = 2
    name = "segment_mean_2"
    sparse_gradient = True
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "num_segments": num_segments, "name": name, "sparse_gradient": sparse_gradient}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    indices = np.array([0, 2, 4], dtype=np.int32)
    segment_ids = np.array([0, 1, 2], dtype=np.int32)
    num_segments = 3
    name = "segment_mean_3"
    sparse_gradient = False
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "num_segments": num_segments, "name": name, "sparse_gradient": sparse_gradient}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    indices = np.array([0, 1, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    num_segments = 2
    name = "segment_mean_4"
    sparse_gradient = True
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "num_segments": num_segments, "name": name, "sparse_gradient": sparse_gradient}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    indices = np.array([0, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 1, 1], dtype=np.int32)
    num_segments = 2
    name = "segment_mean_5"
    sparse_gradient = False
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "num_segments": num_segments, "name": name, "sparse_gradient": sparse_gradient}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 2], dtype=np.int32)
    num_segments = 3
    name = "segment_mean_6"
    sparse_gradient = False
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "num_segments": num_segments, "name": name, "sparse_gradient": sparse_gradient}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: Different data type
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = 2
    name = "segment_mean_7"
    sparse_gradient = True
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "num_segments": num_segments, "name": name, "sparse_gradient": sparse_gradient}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Missing segment ID.
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 2, 3], dtype=np.int32)
    num_segments = 4
    name = "segment_mean_8"
    sparse_gradient = False
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "num_segments": num_segments, "name": name, "sparse_gradient": sparse_gradient}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty data.
    data = np.array([], dtype=np.float32)
    indices = np.array([], dtype=np.int32)
    segment_ids = np.array([], dtype=np.int32)
    num_segments = 1
    name = "segment_mean_9"
    sparse_gradient = True
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "num_segments": num_segments, "name": name, "sparse_gradient": sparse_gradient}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger number of segments
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    segment_ids = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    num_segments = 6
    name = "segment_mean_10"
    sparse_gradient = False
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "num_segments": num_segments, "name": name, "sparse_gradient": sparse_gradient}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.segment_mean"] = tf_sparse_segment_mean_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.segment_mean' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.segment_mean'.")

check_valid('tf.sparse.segment_mean', generated_inputs['tf.sparse.segment_mean'], lib="tf", suffix=0)
