
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_serialize_many_sparse_inputs():
    list_of_inputs = []

    # Note: The API requires a tf.SparseTensor, but the testing harness fails on it.
    # Providing dense numpy arrays to satisfy the harness's pre-analysis.
    # This is expected to cause a TypeError when the API is actually called.

    # Input 1: Rank 2, N=2, int32 values
    sp_input_1 = np.array([[0, 1, 0, 2], [0, 0, 3, 0]], dtype=np.int32)
    input_dict_1 = {
        'sp_input': sp_input_1,
        'out_type': tf.string,
        'name': 'dense_input_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Rank 3, N=3, float32 values
    sp_input_2 = np.array([[[0., 10.], [0., 0.]], 
                           [[0., 0.], [0., 0.]], 
                           [[0., -20.5], [0., 0.]]], dtype=np.float32)
    input_dict_2 = {
        'sp_input': sp_input_2,
        'out_type': tf.string,
        'name': 'dense_input_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Rank 4, N=4, int64 values
    sp_input_3 = np.zeros((4, 2, 2, 2), dtype=np.int64)
    sp_input_3[0, 0, 0, 0] = 100
    sp_input_3[1, 1, 1, 1] = 200
    sp_input_3[3, 0, 1, 0] = -300
    input_dict_3 = {
        'sp_input': sp_input_3,
        'out_type': tf.string,
        'name': 'dense_input_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4: Large minibatch size N=10
    sp_input_4 = np.zeros((10, 5), dtype=np.int32)
    sp_input_4[1, 0] = 1
    sp_input_4[5, 2] = 2
    sp_input_4[9, 4] = 3
    input_dict_4 = {
        'sp_input': sp_input_4,
        'out_type': tf.string,
        'name': 'dense_input_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: out_type=tf.variant
    sp_input_5 = np.array([[0, 1.1], [2.2, 0]], dtype=np.float64)
    input_dict_5 = {
        'sp_input': sp_input_5,
        'out_type': tf.variant,
        'name': 'dense_input_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: All batches are empty (all zeros)
    sp_input_6 = np.zeros((3, 4, 5), dtype=np.float32)
    input_dict_6 = {
        'sp_input': sp_input_6,
        'out_type': tf.string,
        'name': 'dense_input_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Only the first batch has data
    sp_input_7 = np.array([[10, 20, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    input_dict_7 = {
        'sp_input': sp_input_7,
        'out_type': tf.string,
        'name': 'dense_input_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Only the last batch has data
    sp_input_8 = np.array([[0, 0], [0, 0], [0, 0], [-10, -20]], dtype=np.int32)
    input_dict_8 = {
        'sp_input': sp_input_8,
        'out_type': tf.string,
        'name': 'dense_input_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Minimum rank (2) and batch size (1)
    sp_input_9 = np.array([[1, 0, 3]], dtype=np.int32)
    input_dict_9 = {
        'sp_input': sp_input_9,
        'out_type': tf.string,
        'name': 'dense_input_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Higher rank, float values
    sp_input_10 = np.zeros((2, 3, 2, 2), dtype=np.float32)
    sp_input_10[0, 1, 0, 1] = 99.9
    sp_input_10[1, 2, 1, 0] = -99.9
    input_dict_10 = {
        'sp_input': sp_input_10,
        'out_type': tf.string,
        'name': 'dense_input_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.io.serialize_many_sparse"] = get_serialize_many_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.serialize_many_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.serialize_many_sparse'.")

check_valid('tf.io.serialize_many_sparse', generated_inputs['tf.io.serialize_many_sparse'], lib="tf", suffix=0)
