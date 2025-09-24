
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_log_uniform_candidate_sampler_inputs():
    list_of_inputs = []

    # Input 1
    true_classes = np.array([[1, 2]], dtype=np.int64)
    num_true = 2
    num_sampled = 5
    unique = True
    range_max = 10
    seed = 123
    name = "sampler_1"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    true_classes = np.array([[0]], dtype=np.int64)
    num_true = 1
    num_sampled = 3
    unique = False
    range_max = 5
    seed = 42
    name = "sampler_2"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    true_classes = np.array([[100, 200, 300]], dtype=np.int64)
    num_true = 3
    num_sampled = 10
    unique = True
    range_max = 500
    seed = 77
    name = "sampler_3"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    true_classes = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    num_true = 5
    num_sampled = 2
    unique = False
    range_max = 10
    seed = 99
    name = "sampler_4"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - Modified to ensure range_max > max(true_classes)
    true_classes = np.array([[0, 0, 0]], dtype=np.int64)
    num_true = 3
    num_sampled = 4
    unique = True
    range_max = 5
    seed = 101
    name = "sampler_5"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    true_classes = np.array([[1000]], dtype=np.int64)
    num_true = 1
    num_sampled = 1
    unique = True
    range_max = 2000
    seed = 500
    name = "sampler_6"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    true_classes = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)
    num_true = 10
    num_sampled = 15
    unique = True
    range_max = 20
    seed = 707
    name = "sampler_7"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    num_true = 2
    num_sampled = 5
    unique = False
    range_max = 10
    seed = 808
    name = "sampler_8"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    true_classes = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    num_true = 3
    num_sampled = 7
    unique = True
    range_max = 15
    seed = 909
    name = "sampler_9"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    true_classes = np.array([[1]], dtype=np.int64)
    num_true = 1
    num_sampled = 2
    unique = True
    range_max = 3
    seed = 1010
    name = "sampler_10"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.log_uniform_candidate_sampler"] = tf_random_log_uniform_candidate_sampler_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.log_uniform_candidate_sampler' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.log_uniform_candidate_sampler'.")

check_valid('tf.random.log_uniform_candidate_sampler', generated_inputs['tf.random.log_uniform_candidate_sampler'], lib="tf", suffix=0)
