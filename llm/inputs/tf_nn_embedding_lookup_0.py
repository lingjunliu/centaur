
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_embedding_lookup_inputs():
    list_of_inputs = []

    def create_input(params, ids, max_norm, name):
        return {"params": params, "ids": ids, "max_norm": max_norm, "name": name}

    # Input 1
    params = [np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)]
    ids = np.array([0, 1, 2], dtype=np.int32)
    max_norm = None
    name = "embedding_lookup_1"
    list_of_inputs.append(create_input(params, ids, max_norm, name))

    # Input 2
    params = [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), np.array([[7, 8, 9]], dtype=np.float32)]
    ids = np.array([0, 1, 2], dtype=np.int32)
    max_norm = 1.0
    list_of_inputs.append(create_input(params, ids, max_norm, "embedding_lookup_2"))

    # Input 3
    params = [np.array([[1, 2], [3, 4]], dtype=np.float64), np.array([[5, 6], [7, 8]], dtype=np.float64)]
    ids = np.array([0, 1], dtype=np.int64)
    max_norm = 2.0
    list_of_inputs.append(create_input(params, ids, max_norm, "embedding_lookup_3"))

    # Input 4
    params = [np.array([[1, 2, 3, 4]], dtype=np.float32), np.array([[5, 6, 7, 8]], dtype=np.float32)]
    ids = np.array([0, 1], dtype=np.int32)
    max_norm = 0.5
    list_of_inputs.append(create_input(params, ids, max_norm, "embedding_lookup_4"))

    # Input 5
    params = [np.array([[1]], dtype=np.float32)]
    ids = np.array([0], dtype=np.int32)
    max_norm = None
    list_of_inputs.append(create_input(params, ids, max_norm, "embedding_lookup_5"))

    # Input 6
    params = [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)]
    ids = np.array([0, 1, 0], dtype=np.int32)
    max_norm = 1.5
    list_of_inputs.append(create_input(params, ids, max_norm, "embedding_lookup_6"))

    # Input 7
    params = [np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)]
    ids = np.array([0, 2, 3], dtype=np.int64)
    max_norm = None
    list_of_inputs.append(create_input(params, ids, max_norm, "embedding_lookup_7"))

    # Input 8
    params = [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)]
    ids = np.array([[0, 1], [1, 0]], dtype=np.int32)
    max_norm = None
    list_of_inputs.append(create_input(params, ids, max_norm, "embedding_lookup_8"))

    # Input 9
    params = [np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([[5, 6], [7, 8]], dtype=np.float32)]
    ids = np.array([0, 1, 0, 1], dtype=np.int32)
    max_norm = 1.0
    list_of_inputs.append(create_input(params, ids, max_norm, "embedding_lookup_9"))

    # Input 10
    params = [np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.float32)]
    ids = np.array([0, 1], dtype=np.int32)
    max_norm = 5.0
    list_of_inputs.append(create_input(params, ids, max_norm, "embedding_lookup_10"))
    

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
