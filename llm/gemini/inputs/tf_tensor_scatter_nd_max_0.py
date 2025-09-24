
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_tensor_scatter_nd_max_inputs():
    list_of_inputs = []

    # Input 1
    tensor = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    indices = np.array([[1], [4], [5]], dtype=np.int32)
    updates = np.array([1, -1, 1], dtype=np.int32)
    bad_indices_policy = ""
    name = None

    input_dict = {
        "tensor": tensor,
        "indices": indices,
        "updates": updates,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = np.array([[0, 0], [0, 0]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.array([1.0, -1.0], dtype=np.float32)
    bad_indices_policy = ""
    name = "scatter_max"

    input_dict = {
        "tensor": tensor,
        "indices": indices,
        "updates": updates,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=np.int64)
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    updates = np.array([1, -1], dtype=np.int64)
    bad_indices_policy = ""
    name = None

    input_dict = {
        "tensor": tensor,
        "indices": indices,
        "updates": updates,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = np.array([0.0], dtype=np.float64)
    indices = np.array([[0]], dtype=np.int32)
    updates = np.array([1.5], dtype=np.float64)
    bad_indices_policy = ""
    name = "test"

    input_dict = {
        "tensor": tensor,
        "indices": indices,
        "updates": updates,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = np.array([-1, -2, -3], dtype=np.int32)
    indices = np.array([[0], [1], [2]], dtype=np.int32)
    updates = np.array([0, -1, -4], dtype=np.int32)
    bad_indices_policy = ""
    name = None

    input_dict = {
        "tensor": tensor,
        "indices": indices,
        "updates": updates,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([[0, 0], [0, 1]], dtype=np.int32)
    updates = np.array([5, 6], dtype=np.int32)
    bad_indices_policy = ""
    name = None

    input_dict = {
        "tensor": tensor,
        "indices": indices,
        "updates": updates,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([[0], [1], [2]], dtype=np.int32)
    updates = np.array([4.0, 1.0, 5.0], dtype=np.float32)
    bad_indices_policy = ""
    name = None

    input_dict = {
        "tensor": tensor,
        "indices": indices,
        "updates": updates,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = np.array([1, 2, 3], dtype=np.int32)
    indices = np.array([[0], [1], [2]], dtype=np.int64)
    updates = np.array([4, 1, 5], dtype=np.int32)
    bad_indices_policy = ""
    name = None

    input_dict = {
        "tensor": tensor,
        "indices": indices,
        "updates": updates,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    updates = np.array([5, 6], dtype=np.int32)
    bad_indices_policy = ""
    name = None

    input_dict = {
        "tensor": tensor,
        "indices": indices,
        "updates": updates,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    updates = np.array([10, 12], dtype=np.int32)
    bad_indices_policy = ""
    name = None

    input_dict = {
        "tensor": tensor,
        "indices": indices,
        "updates": updates,
        "bad_indices_policy": bad_indices_policy,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.tensor_scatter_nd_max"] = tf_tensor_scatter_nd_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.tensor_scatter_nd_max' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.tensor_scatter_nd_max'.")

check_valid('tf.tensor_scatter_nd_max', generated_inputs['tf.tensor_scatter_nd_max'], lib="tf", suffix=0)
