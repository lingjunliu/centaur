
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
    num_true = 2
    num_sampled = 4
    unique = True
    range_max = 10
    seed = 1
    seed2 = 1
    name = "sampler_1"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    true_classes = np.array([[5, 6, 7], [8, 9, 10]], dtype=np.int64)
    num_true = 3
    num_sampled = 5
    unique = False
    range_max = 15
    seed = 123
    seed2 = 456
    name = "sampler_2"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    true_classes = np.array([[11, 12, 13, 14], [15, 16, 17, 18]], dtype=np.int64)
    num_true = 4
    num_sampled = 6
    unique = True
    range_max = 20
    seed = 789
    seed2 = 101
    name = "sampler_3"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    true_classes = np.array([[1]], dtype=np.int64)
    num_true = 1
    num_sampled = 3
    unique = False
    range_max = 5
    seed = 222
    seed2 = 333
    name = "sampler_4"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    true_classes = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int64)
    num_true = 5
    num_sampled = 7
    unique = True
    range_max = 25
    seed = 444
    seed2 = 555
    name = "sampler_5"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - Different shape for true_classes
    true_classes = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    num_true = 2
    num_sampled = 4
    unique = False
    range_max = 10
    seed = 666
    seed2 = 777
    name = "sampler_6"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7 - Smaller range_max
    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    num_true = 2
    num_sampled = 4
    unique = True
    range_max = 5
    seed = 1
    seed2 = 1
    name = "sampler_7"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    true_classes = np.array([[1,2,3,4,5,6]], dtype=np.int64)
    num_true = 6
    num_sampled = 5
    unique = False
    range_max = 10
    seed = 1
    seed2 = 2
    name = "sampler_8"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    true_classes = np.array([[1,2],[3,4]], dtype=np.int64)
    num_true = 2
    num_sampled = 2
    unique = True
    range_max = 5
    seed = 10
    seed2 = 20
    name = "sampler_9"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    true_classes = np.array([[1]], dtype=np.int64)
    num_true = 1
    num_sampled = 2
    unique = False
    range_max = 3
    seed = 30
    seed2 = 40
    name = "sampler_10"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    #Input 11 - Different num_true
    true_classes = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    num_true = 3
    num_sampled = 5
    unique = True
    range_max = 10
    seed = 50
    seed2 = 60
    name = "sampler_11"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.LearnedUnigramCandidateSampler"] = tf_raw_ops_LearnedUnigramCandidateSampler_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.LearnedUnigramCandidateSampler' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LearnedUnigramCandidateSampler'.")

check_valid('tf.raw_ops.LearnedUnigramCandidateSampler', generated_inputs['tf.raw_ops.LearnedUnigramCandidateSampler'], lib="tf", suffix=0)
