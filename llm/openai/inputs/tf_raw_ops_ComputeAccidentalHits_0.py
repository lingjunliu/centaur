
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ComputeAccidentalHits_inputs():
    list_of_inputs = []

    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([2, 5, 4], dtype=np.int64)
    input_dict = {
        "seed": np.int32(0),
        "seed2": np.int32(0),
        "name": "case_1",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[-1], [0], [7]], dtype=np.int64)
    sampled_candidates = np.array([-1, 8, 9, 10], dtype=np.int64)
    input_dict = {
        "seed": np.int32(123),
        "seed2": np.int32(456),
        "name": "case_2",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[10, 10, 10], [10, 11, 12]], dtype=np.int64)
    sampled_candidates = np.array([10, 11, 12, 13], dtype=np.int64)
    input_dict = {
        "seed": np.int32(999),
        "seed2": np.int32(0),
        "name": "case_3",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[100, 200], [300, 400], [500, 600]], dtype=np.int64)
    sampled_candidates = np.array([700, 800, 900], dtype=np.int64)
    input_dict = {
        "seed": np.int32(0),
        "seed2": np.int32(42),
        "name": "case_4",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([
        [np.int64(np.iinfo(np.int64).min + 1), 0, 5, 999999999999],
        [42, -99999999999, np.int64(np.iinfo(np.int64).max - 1), 7]
    ], dtype=np.int64)
    sampled_candidates = np.array([np.int64(np.iinfo(np.int64).max - 1), 123, 999999999999, -99999999999], dtype=np.int64)
    input_dict = {
        "seed": np.int32(2021),
        "seed2": np.int32(2022),
        "name": "case_5",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(4),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.zeros((5, 1), dtype=np.int64)
    sampled_candidates = np.array([0, 1, 2], dtype=np.int64)
    input_dict = {
        "seed": np.int32(7),
        "seed2": np.int32(8),
        "name": "case_6",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[3, -5], [3, -5], [7, 8]], dtype=np.int64)
    sampled_candidates = np.array([3, -5], dtype=np.int64)
    input_dict = {
        "seed": np.int32(11),
        "seed2": np.int32(12),
        "name": "case_7",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[2, 2, 2, 3, 4]], dtype=np.int64)
    sampled_candidates = np.array([2, 2, 2, 3, 5, 6], dtype=np.int64)
    input_dict = {
        "seed": np.int32(21),
        "seed2": np.int32(22),
        "name": "case_8",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(5),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[10], [20], [30], [40]], dtype=np.int64)
    sampled_candidates = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=np.int64)
    input_dict = {
        "seed": np.int32(100),
        "seed2": np.int32(200),
        "name": "case_9",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.int64)
    sampled_candidates = np.array([6, 5, 4, 3, 2, 1], dtype=np.int64)
    input_dict = {
        "seed": np.int32(31415),
        "seed2": np.int32(27182),
        "name": "case_10",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([4, 0, 2, 6, 8], dtype=np.int64)
    input_dict = {
        "seed": np.int32(555),
        "seed2": np.int32(777),
        "name": "case_11",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1000000000000], [-1000000000000], [5]], dtype=np.int64)
    sampled_candidates = np.array([-1000000000000, 7, 1000000000000], dtype=np.int64)
    input_dict = {
        "seed": np.int32(42),
        "seed2": np.int32(24),
        "name": "case_12",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ComputeAccidentalHits"] = tf_raw_ops_ComputeAccidentalHits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ComputeAccidentalHits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ComputeAccidentalHits'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ComputeAccidentalHits', generated_inputs['tf.raw_ops.ComputeAccidentalHits'], lib="tf", suffix=0)
