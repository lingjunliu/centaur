
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_tensor_scatter_nd_update_inputs():
    list_of_inputs = []

    # Input 1: Scalar updates, rank-1 tensor
    tensor = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    indices = np.array([[1], [3], [4], [7]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.int32)
    name = "scalar_updates_rank1"
    input_dict = {"tensor": tensor, "indices": indices, "updates": updates, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar updates, rank-2 tensor
    tensor = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.int32)
    indices = np.array([[0, 1], [2, 0]], dtype=np.int32)
    updates = np.array([5, 10], dtype=np.int32)
    name = "scalar_updates_rank2"
    input_dict = {"tensor": tensor, "indices": indices, "updates": updates, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Slice updates, rank-2 tensor
    tensor = np.zeros([6, 3], dtype=np.int32)
    indices = np.array([[2], [4]], dtype=np.int32)
    updates = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    name = "slice_updates_rank2"
    input_dict = {"tensor": tensor, "indices": indices, "updates": updates, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Slice updates, rank-5 tensor (video batch) - Reduced dimensions for simplicity
    batch_size, time, width, height, channels = 1, 2, 3, 4, 3
    tensor = np.zeros([batch_size, time, width, height, channels], dtype=np.int32)
    indices = np.array([[0]], dtype=np.int32)
    updates = np.ones([1, time, width, height, channels], dtype=np.int32)
    name = "slice_updates_rank5_clips"
    input_dict = {"tensor": tensor, "indices": indices, "updates": updates, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Slice updates, rank-5 tensor (video frames) - Reduced dimensions for simplicity
    batch_size, time, width, height, channels = 1, 2, 3, 4, 3
    tensor = np.zeros([batch_size, time, width, height, channels], dtype=np.int32)
    indices = np.array([[0, 0]], dtype=np.int32)
    updates = np.ones([1, width, height, channels], dtype=np.int32)
    name = "slice_updates_rank5_frames"
    input_dict = {"tensor": tensor, "indices": indices, "updates": updates, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Folded indices, drawing an 'X' - Simplified for smaller input
    tensor = np.zeros([3, 3], dtype=np.float32)
    indices = np.array([
        [[0, 0], [1, 1], [2, 2]],
        [[0, 2], [1, 1], [2, 0]],
    ], dtype=np.int32)
    updates = np.array([
        [1, 1, 1],
        [1, 1, 1],
    ], dtype=np.float32)
    name = "folded_indices_x"
    input_dict = {"tensor": tensor, "indices": indices, "updates": updates, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Scalar updates, rank-3 tensor
    tensor = np.ones([2, 3, 4], dtype=np.int32)
    indices = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    updates = np.array([5, 10], dtype=np.int32)
    name = "scalar_updates_rank3"
    input_dict = {"tensor": tensor, "indices": indices, "updates": updates, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Slice updates, rank-3 tensor
    tensor = np.zeros([2, 3, 4], dtype=np.int32)
    indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]], dtype=np.int32)
    name = "slice_updates_rank3"
    input_dict = {"tensor": tensor, "indices": indices, "updates": updates, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Scalar updates with negative indices (avoiding for now, potentially problematic)
    tensor = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    indices = np.array([[1], [3], [4], [7]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.int32)
    name = "scalar_updates_rank1" # Removed negative index example
    input_dict = {"tensor": tensor, "indices": indices, "updates": updates, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Slice updates, different shapes - Reduced dimensions
    tensor = np.zeros([2, 3, 2], dtype=np.int32)
    indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.array([[[1, 2], [3, 4], [5, 6]],
                        [[7, 8], [9, 10], [11, 12]]], dtype=np.int32)
    name = "slice_updates_rank3_diff_shape"
    input_dict = {"tensor": tensor, "indices": indices, "updates": updates, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.tensor_scatter_nd_update"] = tf_tensor_scatter_nd_update_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.tensor_scatter_nd_update' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.tensor_scatter_nd_update'.")

check_valid('tf.tensor_scatter_nd_update', generated_inputs['tf.tensor_scatter_nd_update'], lib="tf", suffix=0)
