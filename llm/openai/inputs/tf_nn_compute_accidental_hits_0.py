
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_compute_accidental_hits_inputs():
    list_of_inputs = []
    
    true_classes = np.array([[3], [4]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 5], dtype=np.int64)
    num_true = 1
    seed = 0
    name = "case1_simple_match"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[10, 11], [12, 13], [14, 15]], dtype=np.int64)
    sampled_candidates = np.array([7, 8, 9, 10, 15], dtype=np.int64)
    num_true = 2
    seed = 42
    name = "case2_multi_row_multi_true"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[0, 5, 7]], dtype=np.int64)
    sampled_candidates = np.array([2, 4, 6], dtype=np.int64)
    num_true = 3
    seed = 123
    name = "case3_no_hits"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[0], [1], [2], [3]], dtype=np.int64)
    sampled_candidates = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    num_true = 1
    seed = 999
    name = "case4_all_hit"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[101, 102, 103], [201, 202, 203]], dtype=np.int64)
    sampled_candidates = np.array([0, 102, 200, 202, 300], dtype=np.int64)
    num_true = 3
    seed = 7
    name = "case5_sparse_hits"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[5, 6], [7, 8], [9, 10], [11, 12], [13, 14]], dtype=np.int64)
    sampled_candidates = np.array([1, 3, 5, 8, 10, 12, 14, 16, 18, 20], dtype=np.int64)
    num_true = 2
    seed = 31415
    name = "case6_larger_batch"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[2147483700]], dtype=np.int64)
    sampled_candidates = np.array([2147483700, 9223372000000000000 // 2], dtype=np.int64)
    num_true = 1
    seed = 1
    name = "case7_big_ids"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[2, 4, 6, 8], [1, 3, 5, 7], [0, 9, 10, 11]], dtype=np.int64)
    sampled_candidates = np.array([6, 7, 8, 12, 13, 14], dtype=np.int64)
    num_true = 4
    seed = 12345
    name = "case8_num_true_four"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[100, 200], [300, 400]], dtype=np.int64)
    sampled_candidates = np.array([200, 500, 600, 1000], dtype=np.int64)
    num_true = 2
    seed = 0
    name = "case9_some_hits_with_zero_seed"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([], dtype=np.int64)
    num_true = 2
    seed = 77
    name = "case10_empty_sampled"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.empty((0, 3), dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3], dtype=np.int64)
    num_true = 3
    seed = 222
    name = "case11_empty_batch"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[9, 10, 11, 12, 13]], dtype=np.int64)
    sampled_candidates = np.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=np.int64)
    num_true = 5
    seed = 555
    name = "case12_single_row_many_true"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.compute_accidental_hits"] = tf_nn_compute_accidental_hits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.compute_accidental_hits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.compute_accidental_hits'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.compute_accidental_hits', generated_inputs['tf.nn.compute_accidental_hits'], lib="tf", suffix=0)
