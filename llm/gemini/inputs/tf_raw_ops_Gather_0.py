
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_gather_inputs():
    list_of_inputs = []

    # Input 1
    params = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    indices = np.array([0, 2, 4], dtype=np.int32)
    validate_indices = True
    name = "gather_example_1"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    validate_indices = False
    name = "gather_example_2"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([[0, 1], [1, 2]], dtype=np.int32)
    validate_indices = True
    name = "gather_example_3"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    validate_indices = False
    name = "gather_example_4"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    params = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    indices = np.array([0, 2, 4], dtype=np.int64)
    validate_indices = True
    name = "gather_example_5"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    indices = np.array([0, 1, 2], dtype=np.int64)
    validate_indices = False
    name = "gather_example_6"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
    validate_indices = True
    name = "gather_example_7"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int64)
    validate_indices = False
    name = "gather_example_8"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger indices and params
    params = np.arange(100, dtype=np.int32).reshape(10, 10)
    indices = np.array([1, 5, 9, 0], dtype=np.int32)
    validate_indices = True
    name = "gather_example_9"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-dimensional indices
    params = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    validate_indices = False
    name = "gather_example_10"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Single element indices
    params = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    indices = np.array(2, dtype=np.int32)
    validate_indices = True
    name = "gather_example_11"
    input_dict = {"params": params, "indices": indices, "validate_indices": validate_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Gather"] = tf_raw_ops_gather_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Gather' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Gather'.")

check_valid('tf.raw_ops.Gather', generated_inputs['tf.raw_ops.Gather'], lib="tf", suffix=0)
