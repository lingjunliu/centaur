
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_gather_v2_inputs():
    list_of_inputs = []

    # Input 1
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    batch_dims = 0
    name = "gather_example_1"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    axis = np.array(1, dtype=np.int32)
    batch_dims = 0
    name = "gather_example_2"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    params = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    batch_dims = 0
    name = "gather_example_3"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    params = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    indices = np.array([2, 4], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    batch_dims = 0
    name = "gather_example_4"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    params = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    axis = np.array(-1, dtype=np.int32)
    batch_dims = 0
    name = "gather_example_5"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    axis = np.array(-2, dtype=np.int32)
    batch_dims = 0
    name = "gather_example_6"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    params = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    indices = np.array([[0, 1], [2, 0]], dtype=np.int32)
    axis = np.array(1, dtype=np.int32)
    batch_dims = 0
    name = "gather_example_7"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64 indices
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int64)
    axis = np.array(0, dtype=np.int32)
    batch_dims = 0
    name = "gather_example_8"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Batch dims - Corrected
    params = np.arange(2 * 3 * 2).reshape((2, 3, 2)).astype(np.int32)
    indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    axis = np.array(1, dtype=np.int32)
    batch_dims = 1
    name = "gather_example_9"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int16 indices
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int16)
    axis = np.array(0, dtype=np.int32)
    batch_dims = 0
    name = "gather_example_10"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.GatherV2"] = tf_raw_ops_gather_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.GatherV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GatherV2'.")

check_valid('tf.raw_ops.GatherV2', generated_inputs['tf.raw_ops.GatherV2'], lib="tf", suffix=0)
