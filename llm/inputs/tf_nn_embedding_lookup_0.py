
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_nn_embedding_lookup_inputs():
    """
    Generates a list of valid inputs for tf.nn.embedding_lookup.
    """
    list_of_inputs = []

    # Input 1: Basic case with a single params tensor
    params_1 = np.random.rand(10, 3).astype(np.float32)
    ids_1 = np.array([1, 5, 9, 2], dtype=np.int32)
    input_dict_1 = {
        'params': params_1,
        'ids': ids_1,
        'max_norm': 1.0,
        'name': 'basic_lookup'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Replacing sharded with a single large tensor
    params_2 = np.random.rand(10, 4).astype(np.float32)
    ids_2 = np.array([0, 1, 6, 8, 4], dtype=np.int64)
    input_dict_2 = {
        'params': params_2,
        'ids': ids_2,
        'max_norm': 2.5,
        'name': 'single_tensor_lookup'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Higher-dimensional embeddings
    params_3 = np.random.rand(8, 2, 3).astype(np.float64)
    ids_3 = np.array([7, 0, 3], dtype=np.int32)
    input_dict_3 = {
        'params': params_3,
        'ids': ids_3,
        'max_norm': 100.0,
        'name': 'high_dim_embedding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 2D ids tensor
    params_4 = np.random.rand(20, 5).astype(np.float32)
    ids_4 = np.array([[0, 2, 4], [10, 12, 14]], dtype=np.int32)
    input_dict_4 = {
        'params': params_4,
        'ids': ids_4,
        'max_norm': 0.5,
        'name': '2d_ids_lookup'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Another single tensor case (replacing uneven sharding)
    params_5 = np.random.rand(8, 10).astype(np.float32)
    ids_5 = np.array([0, 3, 6, 7], dtype=np.int64)
    input_dict_5 = {
        'params': params_5,
        'ids': ids_5,
        'max_norm': 1.5,
        'name': 'single_tensor_uneven_ids'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: float64 data type for params
    params_6 = np.random.rand(8, 8).astype(np.float64)
    ids_6 = np.array([1, 3, 5, 7], dtype=np.int64)
    input_dict_6 = {
        'params': params_6,
        'ids': ids_6,
        'max_norm': 5.0,
        'name': 'float64_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 2D ids with a single tensor (replacing sharded case)
    params_7 = np.random.rand(20, 2).astype(np.float32)
    ids_7 = np.array([[1, 11], [5, 15], [9, 19]], dtype=np.int32)
    input_dict_7 = {
        'params': params_7,
        'ids': ids_7,
        'max_norm': 1.2,
        'name': 'single_tensor_2d_ids'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 1D embeddings (embedding vectors are scalars)
    params_8 = np.random.rand(12, 1).astype(np.float32)
    ids_8 = np.array([11, 10, 1, 0, 5], dtype=np.int32)
    input_dict_8 = {
        'params': params_8,
        'ids': ids_8,
        'max_norm': 10.0,
        'name': 'scalar_embeddings'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Empty ids tensor
    params_9 = np.random.rand(5, 5).astype(np.float32)
    ids_9 = np.array([], dtype=np.int32)
    input_dict_9 = {
        'params': params_9,
        'ids': ids_9,
        'max_norm': 1.0,
        'name': 'empty_ids'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Repeated ids
    params_10 = np.random.rand(8, 3).astype(np.float32)
    ids_10 = np.array([0, 5, 0, 5, 1, 6, 1], dtype=np.int64)
    input_dict_10 = {
        'params': params_10,
        'ids': ids_10,
        'max_norm': 3.0,
        'name': 'repeated_ids'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: float16 data type for params
    params_11 = np.random.rand(10, 4).astype(np.float16)
    ids_11 = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    input_dict_11 = {
        'params': params_11,
        'ids': ids_11,
        'max_norm': 1.0,
        'name': 'float16_params'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: max_norm is None
    params_12 = np.random.rand(5, 5).astype(np.float32)
    ids_12 = np.array([[0,1],[2,3]], dtype=np.int32)
    input_dict_12 = {
        'params': params_12,
        'ids': ids_12,
        'max_norm': None,
        'name': 'no_max_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))


    return list_of_inputs

generated_inputs["tf.nn.embedding_lookup"] = tf_nn_embedding_lookup_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.embedding_lookup' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.embedding_lookup'.")

check_valid('tf.nn.embedding_lookup', generated_inputs['tf.nn.embedding_lookup'], lib="tf", suffix=0)
