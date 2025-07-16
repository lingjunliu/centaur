
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_embedding_lookup_inputs():
    list_of_inputs = []

    # Input 1: Single tensor, simple case
    params = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)
    ids = np.array([0, 3, 4], dtype=np.int32)
    max_norm = None
    name = "embedding_lookup_1"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of tensors, basic case
    params_list = [np.array([[1, 2], [3, 4]], dtype=np.float32),
                   np.array([[5, 6], [7, 8]], dtype=np.float32),
                   np.array([[9, 10]], dtype=np.float32)]
    ids = np.array([0, 3, 4], dtype=np.int32)
    max_norm = None
    name = "embedding_lookup_2"

    input_dict = {
        "params": params_list,
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different ids
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    ids = np.array([2, 1, 0], dtype=np.int32)
    max_norm = None
    name = "embedding_lookup_3"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With max_norm
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    ids = np.array([0, 1, 2], dtype=np.int32)
    max_norm = 3.0
    name = "embedding_lookup_4"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Higher dimension params
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    ids = np.array([0, 1], dtype=np.int32)
    max_norm = None
    name = "embedding_lookup_5"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64 ids
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    ids = np.array([0, 1, 2], dtype=np.int64)
    max_norm = None
    name = "embedding_lookup_6"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List of tensors, more partitions
    params_list = [np.array([[1, 2]], dtype=np.float32),
                   np.array([[3, 4]], dtype=np.float32),
                   np.array([[5, 6]], dtype=np.float32),
                   np.array([[7, 8]], dtype=np.float32),
                   np.array([[9, 10]], dtype=np.float32)]
    ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    max_norm = None
    name = "embedding_lookup_7"

    input_dict = {
        "params": params_list,
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty ids array
    params = np.array([[1, 2], [3, 4]], dtype=np.float32)
    ids = np.array([], dtype=np.int32)
    max_norm = None
    name = "embedding_lookup_8"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single tensor, float64 params
    params = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float64)
    ids = np.array([0, 3, 4], dtype=np.int32)
    max_norm = None
    name = "embedding_lookup_9"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: max_norm is a number
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    ids = np.array([0, 1, 2], dtype=np.int32)
    max_norm = 1.0
    name = "embedding_lookup_10"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: more complex max_norm
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    ids = np.array([0, 1, 2], dtype=np.int32)
    max_norm = 5.0
    name = "embedding_lookup_11"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
my_inputs = tf_nn_embedding_lookup_inputs()
generated_inputs["tf.nn.embedding_lookup"] = my_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.embedding_lookup' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.embedding_lookup'.")

check_valid('tf.nn.embedding_lookup', generated_inputs['tf.nn.embedding_lookup'], lib="tf", suffix=0)
