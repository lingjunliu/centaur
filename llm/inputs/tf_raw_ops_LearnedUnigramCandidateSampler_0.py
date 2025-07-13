
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_LearnedUnigramCandidateSampler_inputs():
    list_of_inputs = []

    # Input 1
    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    num_true = 1
    num_sampled = 3
    unique = True
    range_max = 10
    seed = 0
    seed2 = 0
    name = "test_sampler_1"
    input_dict = {
        "true_classes": true_classes,
        "num_true": np.int32(num_true),
        "num_sampled": np.int32(num_sampled),
        "unique": np.bool_(unique),
        "range_max": np.int32(range_max),
        "seed": np.int32(seed),
        "seed2": np.int32(seed2),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    true_classes = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    num_true = 2
    num_sampled = 5
    unique = False
    range_max = 20
    seed = 123
    seed2 = 456
    name = "test_sampler_2"
    input_dict = {
        "true_classes": true_classes,
        "num_true": np.int32(num_true),
        "num_sampled": np.int32(num_sampled),
        "unique":  np.bool_(unique),
        "range_max": np.int32(range_max),
        "seed": np.int32(seed),
        "seed2": np.int32(seed2),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    true_classes = np.array([[1]], dtype=np.int64)
    num_true = 1
    num_sampled = 1
    unique = True
    range_max = 2
    seed = 789
    seed2 = 101
    name = "test_sampler_3"
    input_dict = {
        "true_classes": true_classes,
        "num_true": np.int32(num_true),
        "num_sampled": np.int32(num_sampled),
        "unique":  np.bool_(unique),
        "range_max": np.int32(range_max),
        "seed": np.int32(seed),
        "seed2": np.int32(seed2),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    true_classes = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    num_true = 3
    num_sampled = 7
    unique = False
    range_max = 30
    seed = 0
    seed2 = 1
    name = "test_sampler_4"
    input_dict = {
        "true_classes": true_classes,
        "num_true": np.int32(num_true),
        "num_sampled": np.int32(num_sampled),
        "unique":  np.bool_(unique),
        "range_max": np.int32(range_max),
        "seed": np.int32(seed),
        "seed2": np.int32(seed2),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    true_classes = np.array([[100, 200], [300, 400]], dtype=np.int64)
    num_true = 1
    num_sampled = 3
    unique = True
    range_max = 500
    seed = 10
    seed2 = 20
    name = "test_sampler_5"
    input_dict = {
        "true_classes": true_classes,
        "num_true": np.int32(num_true),
        "num_sampled": np.int32(num_sampled),
        "unique":  np.bool_(unique),
        "range_max": np.int32(range_max),
        "seed": np.int32(seed),
        "seed2": np.int32(seed2),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    true_classes = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int64)
    num_true = 4
    num_sampled = 9
    unique = False
    range_max = 40
    seed = 30
    seed2 = 40
    name = "test_sampler_6"
    input_dict = {
        "true_classes": true_classes,
        "num_true": np.int32(num_true),
        "num_sampled": np.int32(num_sampled),
        "unique":  np.bool_(unique),
        "range_max": np.int32(range_max),
        "seed": np.int32(seed),
        "seed2": np.int32(seed2),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    true_classes = np.array([[1000]], dtype=np.int64)
    num_true = 1
    num_sampled = 2
    unique = True
    range_max = 2000
    seed = 50
    seed2 = 60
    name = "test_sampler_7"
    input_dict = {
        "true_classes": true_classes,
        "num_true": np.int32(num_true),
        "num_sampled": np.int32(num_sampled),
        "unique":  np.bool_(unique),
        "range_max": np.int32(range_max),
        "seed": np.int32(seed),
        "seed2": np.int32(seed2),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    true_classes = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int64)
    num_true = 2
    num_sampled = 6
    unique = False
    range_max = 50
    seed = 70
    seed2 = 80
    name = "test_sampler_8"
    input_dict = {
        "true_classes": true_classes,
        "num_true": np.int32(num_true),
        "num_sampled": np.int32(num_sampled),
        "unique":  np.bool_(unique),
        "range_max": np.int32(range_max),
        "seed": np.int32(seed),
        "seed2": np.int32(seed2),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    true_classes = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    num_true = 1
    num_sampled = 4
    unique = True
    range_max = 15
    seed = 90
    seed2 = 100
    name = "test_sampler_9"
    input_dict = {
        "true_classes": true_classes,
        "num_true": np.int32(num_true),
        "num_sampled": np.int32(num_sampled),
        "unique":  np.bool_(unique),
        "range_max": np.int32(range_max),
        "seed": np.int32(seed),
        "seed2": np.int32(seed2),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    true_classes = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    num_true = 5
    num_sampled = 11
    unique = False
    range_max = 60
    seed = 110
    seed2 = 120
    name = "test_sampler_10"
    input_dict = {
        "true_classes": true_classes,
        "num_true": np.int32(num_true),
        "num_sampled": np.int32(num_sampled),
        "unique":  np.bool_(unique),
        "range_max": np.int32(range_max),
        "seed": np.int32(seed),
        "seed2": np.int32(seed2),
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_LearnedUnigramCandidateSampler_inputs()
generated_inputs["tf.raw_ops.LearnedUnigramCandidateSampler"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.LearnedUnigramCandidateSampler"].append({
        "args": [],
        "kwargs": input_dict
    })

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.LearnedUnigramCandidateSampler' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LearnedUnigramCandidateSampler'.")

check_valid('tf.raw_ops.LearnedUnigramCandidateSampler', generated_inputs['tf.raw_ops.LearnedUnigramCandidateSampler'], lib="tf", suffix=0)
