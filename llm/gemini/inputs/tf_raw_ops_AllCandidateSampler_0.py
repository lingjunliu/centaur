
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_all_candidate_sampler_inputs():
    list_of_inputs = []

    # Input 1
    true_classes = np.array([[0, 1]], dtype=np.int64)
    num_true = 2
    num_sampled = 5
    unique = True
    seed = 1
    seed2 = 2
    name = None
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    true_classes = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
    num_true = 3
    num_sampled = 3
    unique = False
    seed = 123
    seed2 = 456
    name = "sampler"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    true_classes = np.array([[0], [1], [2]], dtype=np.int64)
    num_true = 1
    num_sampled = 10
    unique = True
    seed = 789
    seed2 = 101
    name = None
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    true_classes = np.array([[0, 1, 2, 3]], dtype=np.int64)
    num_true = 4
    num_sampled = 7
    unique = False
    seed = 1000
    seed2 = 1
    name = "test_sampler"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    true_classes = np.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.int64)
    num_true = 2
    num_sampled = 2
    unique = True
    seed = 10
    seed2 = 20
    name = None
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    true_classes = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
    num_true = 5
    num_sampled = 8
    unique = False
    seed = 55
    seed2 = 66
    name = "sampler_6"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    true_classes = np.array([[0]], dtype=np.int64)
    num_true = 1
    num_sampled = 1
    unique = True
    seed = 111
    seed2 = 222
    name = None
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    true_classes = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)
    num_true = 6
    num_sampled = 9
    unique = False
    seed = 777
    seed2 = 888
    name = "test8"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    true_classes = np.array([[0, 1], [0, 1]], dtype=np.int64)
    num_true = 2
    num_sampled = 4
    unique = True
    seed = 444
    seed2 = 555
    name = None
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    true_classes = np.array([[0, 1, 2, 3, 4, 5, 6]], dtype=np.int64)
    num_true = 7
    num_sampled = 10
    unique = False
    seed = 999
    seed2 = 12
    name = "sampler_10"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AllCandidateSampler"] = tf_raw_ops_all_candidate_sampler_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AllCandidateSampler' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AllCandidateSampler'.")

check_valid('tf.raw_ops.AllCandidateSampler', generated_inputs['tf.raw_ops.AllCandidateSampler'], lib="tf", suffix=0)
