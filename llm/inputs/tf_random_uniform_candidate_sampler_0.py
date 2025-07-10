
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_uniform_candidate_sampler_inputs():
    list_of_inputs = []

    # Input 1
    true_classes = np.array([[1, 2]], dtype=np.int64)
    num_true = 2
    num_sampled = 3
    unique = True
    range_max = 10
    seed = 123
    name = "uniform_sampler_1"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    true_classes = np.array([[0], [1], [2]], dtype=np.int64)
    num_true = 1
    num_sampled = 5
    unique = False
    range_max = 5
    seed = 456
    name = "uniform_sampler_2"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    true_classes = np.array([[1, 2, 3, 4]], dtype=np.int64)
    num_true = 4
    num_sampled = 7
    unique = True
    range_max = 15
    seed = 789
    name = "uniform_sampler_3"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    true_classes = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
    num_true = 2
    num_sampled = 2
    unique = True
    range_max = 6
    seed = 101
    name = "uniform_sampler_4"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    true_classes = np.array([[10, 11, 12]], dtype=np.int64)
    num_true = 3
    num_sampled = 4
    unique = False
    range_max = 20
    seed = 112
    name = "uniform_sampler_5"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    true_classes = np.array([[0]], dtype=np.int64)
    num_true = 1
    num_sampled = 1
    unique = True
    range_max = 2
    seed = 123
    name = "uniform_sampler_6"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    true_classes = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    num_true = 5
    num_sampled = 2
    unique = True
    range_max = 8
    seed = 789
    name = "uniform_sampler_7"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    true_classes = np.array([[0, 1, 2, 3]], dtype=np.int64)
    num_true = 4
    num_sampled = 4
    unique = False
    range_max = 7
    seed = 101
    name = "uniform_sampler_8"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    true_classes = np.array([[10, 11]], dtype=np.int64)
    num_true = 2
    num_sampled = 6
    unique = True
    range_max = 12
    seed = 112
    name = "uniform_sampler_9"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    true_classes = np.array([[5]], dtype=np.int64)
    num_true = 1
    num_sampled = 1
    unique = False
    range_max = 10
    seed = 123
    name = "uniform_sampler_10"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.uniform_candidate_sampler"] = tf_random_uniform_candidate_sampler_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.uniform_candidate_sampler' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.uniform_candidate_sampler'.")

check_valid('tf.random.uniform_candidate_sampler', generated_inputs['tf.random.uniform_candidate_sampler'], lib="tf", suffix=0)
