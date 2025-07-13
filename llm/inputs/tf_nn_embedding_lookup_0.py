
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_embedding_lookup_inputs():
    list_of_inputs = []

    # Input 1: Single params tensor
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

    # Input 2: List of params tensors
    params1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    params2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
    params3 = np.array([[9, 10]], dtype=np.float32)
    ids = np.array([0, 1, 0], dtype=np.int32)
    max_norm = 1.0
    name = "embedding_lookup_2"

    input_dict = {
        "params": [params1, params2, params3],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different ids
    params = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    ids = np.array([0, 1, 2, 0], dtype=np.int32)
    max_norm = None
    name = "embedding_lookup_3"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shape of params
    params = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    ids = np.array([0, 1], dtype=np.int32)
    max_norm = None
    name = "embedding_lookup_4"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: max_norm provided
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    ids = np.array([0, 1, 2], dtype=np.int32)
    max_norm = 5.0
    name = "embedding_lookup_5"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: ids with different dtype (int64)
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    ids = np.array([0, 1, 2], dtype=np.int64)
    max_norm = 5.0
    name = "embedding_lookup_6"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: 2D ids
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    ids = np.array([[0, 1], [2, 0]], dtype=np.int32)
    max_norm = None
    name = "embedding_lookup_7"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger embedding dimension
    params = np.random.rand(10, 20).astype(np.float32)
    ids = np.array([1, 5, 9], dtype=np.int32)
    max_norm = None
    name = "embedding_lookup_8"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: List of tensors, larger ids, adjust ids range
    params1 = np.random.rand(5, 10).astype(np.float32)
    params2 = np.random.rand(5, 10).astype(np.float32)
    ids = np.array([0, 1, 2, 3], dtype=np.int32)
    max_norm = 1.5
    name = "embedding_lookup_9"

    input_dict = {
        "params": [params1, params2],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D ids. Reduced the ID range to be valid.
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    ids = np.array([[[0, 1]], [[1, 0]]], dtype=np.int32)
    max_norm = None
    name = "embedding_lookup_10"

    input_dict = {
        "params": [params],
        "ids": ids,
        "max_norm": max_norm,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.embedding_lookup"] = tf_nn_embedding_lookup_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.embedding_lookup' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.embedding_lookup'.")

check_valid('tf.nn.embedding_lookup', generated_inputs['tf.nn.embedding_lookup'], lib="tf", suffix=0)
