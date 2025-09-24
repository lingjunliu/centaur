
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_gather_nd_inputs():
    list_of_inputs = []

    # Input 1
    params = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    batch_dims = 0
    name = "gather_example_1"
    bad_indices_policy = ""
    input_dict = {"params": params, "indices": indices, "batch_dims": batch_dims, "name": name, "bad_indices_policy": bad_indices_policy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    batch_dims = 0
    name = "gather_example_2"
    bad_indices_policy = ""
    input_dict = {"params": params, "indices": indices, "batch_dims": batch_dims, "name": name, "bad_indices_policy": bad_indices_policy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    params = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array([[1], [0]], dtype=np.int32)
    batch_dims = 0
    name = "gather_example_3"
    bad_indices_policy = ""
    input_dict = {"params": params, "indices": indices, "batch_dims": batch_dims, "name": name, "bad_indices_policy": bad_indices_policy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([[[1]], [[0]]], dtype=np.int32)
    batch_dims = 0
    name = "gather_example_4"
    bad_indices_policy = ""
    input_dict = {"params": params, "indices": indices, "batch_dims": batch_dims, "name": name, "bad_indices_policy": bad_indices_policy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([[[0, 1], [1, 0]], [[0, 0], [1, 1]]], dtype=np.int32)
    batch_dims = 0
    name = "gather_example_5"
    bad_indices_policy = ""
    input_dict = {"params": params, "indices": indices, "batch_dims": batch_dims, "name": name, "bad_indices_policy": bad_indices_policy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batched params and indices
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([[1], [0]], dtype=np.int32)
    batch_dims = 1
    name = "gather_example_6"
    bad_indices_policy = ""
    input_dict = {"params": params, "indices": indices, "batch_dims": batch_dims, "name": name, "bad_indices_policy": bad_indices_policy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batched params and indices, different indices
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([[[1, 0]], [[0, 1]]], dtype=np.int32)
    batch_dims = 1
    name = "gather_example_7"
    bad_indices_policy = ""
    input_dict = {"params": params, "indices": indices, "batch_dims": batch_dims, "name": name, "bad_indices_policy": bad_indices_policy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bad_indices_policy="IGNORE", valid indices
    params = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    batch_dims = 0
    name = "gather_example_8"
    bad_indices_policy = "IGNORE"
    input_dict = {"params": params, "indices": indices, "batch_dims": batch_dims, "name": name, "bad_indices_policy": bad_indices_policy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64 indices
    params = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    batch_dims = 0
    name = "gather_example_9"
    bad_indices_policy = ""
    input_dict = {"params": params, "indices": indices, "batch_dims": batch_dims, "name": name, "bad_indices_policy": bad_indices_policy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More complex shape for params
    params = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    indices = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32)
    batch_dims = 0
    name = "gather_example_10"
    bad_indices_policy = ""
    input_dict = {"params": params, "indices": indices, "batch_dims": batch_dims, "name": name, "bad_indices_policy": bad_indices_policy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: batch_dims > 0, more complex shape
    params = np.arange(30, dtype=np.int32).reshape((2, 3, 5))
    indices = np.array([[0, 1], [1, 2]], dtype=np.int32)
    batch_dims = 1
    name = "gather_example_11"
    bad_indices_policy = ""
    input_dict = {"params": params, "indices": indices, "batch_dims": batch_dims, "name": name, "bad_indices_policy": bad_indices_policy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.gather_nd"] = tf_gather_nd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.gather_nd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.gather_nd'.")

check_valid('tf.gather_nd', generated_inputs['tf.gather_nd'], lib="tf", suffix=0)
