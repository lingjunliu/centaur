
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_embedding_lookup_inputs():
    list_of_inputs = []

    def create_input_dict(params, ids, max_norm, name):
        return {"params": [params], "ids": ids, "max_norm": max_norm, "name": name}

    # Input 1: Basic example
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    ids = np.array([0, 1, 2], dtype=np.int32)
    input_dict = create_input_dict(params, ids, None, "basic_example")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D ids
    params = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    ids = np.array([[0, 1], [2, 3]], dtype=np.int32)
    input_dict = create_input_dict(params, ids, None, "2d_ids")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: max_norm clipping
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    ids = np.array([0, 1, 2], dtype=np.int32)
    max_norm = 4.0
    input_dict = create_input_dict(params, ids, max_norm, "max_norm")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different dtype for ids
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    ids = np.array([0, 1, 2], dtype=np.int64)
    input_dict = create_input_dict(params, ids, None, "int64_ids")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D params
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    ids = np.array([0, 1, 2], dtype=np.int32)
    input_dict = create_input_dict(params, ids, None, "3d_params")
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Empty IDs
    params = np.array([[1, 2], [3, 4]], dtype=np.float32)
    ids = np.array([], dtype=np.int32)
    input_dict = create_input_dict(params, ids, None, "empty_ids")
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single ID
    params = np.array([[1, 2], [3, 4]], dtype=np.float32)
    ids = np.array(0, dtype=np.int32)
    input_dict = create_input_dict(params, ids, None, "single_id")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List of params, more than one id per partition
    params1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    params2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
    params3 = np.array([[9, 10], [11, 12]], dtype=np.float32)
    ids = np.array([0, 2, 4], dtype=np.int32)

    input_dict = {"params": [params1, params2, params3], "ids": ids, "max_norm": None, "name": "list_params_multi_id"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: max_norm and list params
    params1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    params2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
    ids = np.array([0, 1, 2, 3], dtype=np.int32)
    max_norm = 6.0
    input_dict = {"params": [params1, params2], "ids": ids, "max_norm": max_norm, "name": "list_params_max_norm"}
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
