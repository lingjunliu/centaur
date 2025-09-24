
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_compute_accidental_hits_inputs():
    list_of_inputs = []

    # Input 1
    true_classes = np.array([[1], [2], [3]], dtype=np.int64)
    sampled_candidates = np.array([1, 4, 2, 5, 3], dtype=np.int64)
    num_true = 1
    seed = 0
    seed2 = 0
    name = "test1"

    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    num_true = 2
    seed = 123
    seed2 = 456
    name = "test2"

    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    true_classes = np.array([[1]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3], dtype=np.int64)
    num_true = 1
    seed = 0
    seed2 = 1
    name = "test3"

    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    true_classes = np.array([[1, 2, 3]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    num_true = 3
    seed = 7
    seed2 = 8
    name = "test4"

    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    true_classes = np.array([[1], [2], [3], [4], [5]], dtype=np.int64)
    sampled_candidates = np.array([6, 7, 8, 9, 10], dtype=np.int64)
    num_true = 1
    seed = 9
    seed2 = 10
    name = "test5"

    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    true_classes = np.array([[1], [1], [1]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    num_true = 1
    seed = 11
    seed2 = 12
    name = "test6"

    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    true_classes = np.array([[1], [2]], dtype=np.int64)
    sampled_candidates = np.array([1, 2], dtype=np.int64)
    num_true = 1
    seed = 13
    seed2 = 14
    name = "test7"

    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    true_classes = np.array([[1], [2]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4], dtype=np.int64)
    num_true = 1
    seed = 15
    seed2 = 16
    name = "test8"

    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    true_classes = np.array([[1], [2], [3], [4]], dtype=np.int64)
    sampled_candidates = np.array([5, 6, 7, 8], dtype=np.int64)
    num_true = 1
    seed = 17
    seed2 = 18
    name = "test9"

    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    true_classes = np.array([[1]], dtype=np.int64)
    sampled_candidates = np.array([1], dtype=np.int64)
    num_true = 1
    seed = 19
    seed2 = 20
    name = "test10"

    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ComputeAccidentalHits"] = tf_raw_ops_compute_accidental_hits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ComputeAccidentalHits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ComputeAccidentalHits'.")

check_valid('tf.raw_ops.ComputeAccidentalHits', generated_inputs['tf.raw_ops.ComputeAccidentalHits'], lib="tf", suffix=0)
