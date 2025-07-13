
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_gather_nd_inputs():
    list_of_inputs = []

    # Input 1
    params = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    bad_indices_policy = ""
    name = None
    input_dict = {"params": params, "indices": indices, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    params = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([[1], [0]], dtype=np.int32)
    bad_indices_policy = ""
    name = None
    input_dict = {"params": params, "indices": indices, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([[1]], dtype=np.int32)
    bad_indices_policy = ""
    name = None
    input_dict = {"params": params, "indices": indices, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    bad_indices_policy = ""
    name = None
    input_dict = {"params": params, "indices": indices, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([[0, 0, 1], [1, 0, 1]], dtype=np.int32)
    bad_indices_policy = ""
    name = None
    input_dict = {"params": params, "indices": indices, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    params = np.array([['a', 'b'], ['c', 'd']], dtype=np.str_)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    bad_indices_policy = ""
    name = None
    input_dict = {"params": params, "indices": indices, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    params = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    indices = np.array([[0], [2], [4]], dtype=np.int32)
    bad_indices_policy = ""
    name = None
    input_dict = {"params": params, "indices": indices, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    params = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int16)
    bad_indices_policy = "IGNORE"
    name = None
    input_dict = {"params": params, "indices": indices, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    params = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    bad_indices_policy = "DEFAULT"
    name = None
    input_dict = {"params": params, "indices": indices, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    params = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([[1, 0], [0, 1]], dtype=np.int32) # Valid indices
    bad_indices_policy = "IGNORE"
    name = None
    input_dict = {"params": params, "indices": indices, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.GatherNd"] = tf_raw_ops_gather_nd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.GatherNd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GatherNd'.")

check_valid('tf.raw_ops.GatherNd', generated_inputs['tf.raw_ops.GatherNd'], lib="tf", suffix=0)
