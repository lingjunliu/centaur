
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_scatter_nd_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.int32)
    shape = np.array([8], dtype=np.int32)
    bad_indices_policy = ""
    name = None
    input_dict = {"indices": indices, "updates": updates, "shape": shape, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.array([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                         [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]], dtype=np.int32)
    shape = np.array([4, 4, 4], dtype=np.int32)
    bad_indices_policy = ""
    name = None
    input_dict = {"indices": indices, "updates": updates, "shape": shape, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Higher rank indices and updates - FIXED SHAPE ISSUE
    indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    updates = np.array([1, 4], dtype=np.int32)
    shape = np.array([2, 2], dtype=np.int32)
    bad_indices_policy = ""
    name = None
    input_dict = {"indices": indices, "updates": updates, "shape": shape, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type for updates
    indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([3.14, 2.71], dtype=np.float32)
    shape = np.array([5], dtype=np.int32)
    bad_indices_policy = ""
    name = None
    input_dict = {"indices": indices, "updates": updates, "shape": shape, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional updates
    indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    shape = np.array([3, 3], dtype=np.int32)
    bad_indices_policy = ""
    name = None
    input_dict = {"indices": indices, "updates": updates, "shape": shape, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty updates
    indices = np.array([], dtype=np.int32).reshape(0, 1)
    updates = np.array([], dtype=np.int32)
    shape = np.array([5], dtype=np.int32)
    bad_indices_policy = ""
    name = None
    input_dict = {"indices": indices, "updates": updates, "shape": shape, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero shape
    indices = np.array([], dtype=np.int32).reshape(0, 1)
    updates = np.array([], dtype=np.int32)
    shape = np.array([0], dtype=np.int32)
    bad_indices_policy = ""
    name = None
    input_dict = {"indices": indices, "updates": updates, "shape": shape, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bad_indices_policy = "ERROR"
    indices = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.int32)
    shape = np.array([8], dtype=np.int32)
    bad_indices_policy = "ERROR"
    name = None
    input_dict = {"indices": indices, "updates": updates, "shape": shape, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: bad_indices_policy = "IGNORE" with out-of-bounds index
    indices = np.array([[4], [3], [1], [9]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.int32)
    shape = np.array([8], dtype=np.int32)
    bad_indices_policy = "IGNORE"
    name = None
    input_dict = {"indices": indices, "updates": updates, "shape": shape, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Name specified
    indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([3.14, 2.71], dtype=np.float32)
    shape = np.array([5], dtype=np.int32)
    bad_indices_policy = ""
    name = "my_scatter"
    input_dict = {"indices": indices, "updates": updates, "shape": shape, "bad_indices_policy": bad_indices_policy, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.scatter_nd"] = tf_scatter_nd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.scatter_nd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.scatter_nd'.")

check_valid('tf.scatter_nd', generated_inputs['tf.scatter_nd'], lib="tf", suffix=0)
