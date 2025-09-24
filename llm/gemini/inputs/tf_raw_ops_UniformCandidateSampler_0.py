
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_UniformCandidateSampler_inputs():
    list_of_inputs = []

    # Input 1
    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    num_true = 2
    num_sampled = 4
    unique = True
    range_max = 10
    seed = 1
    seed2 = 1
    name = "uniform_candidate_sampler_1"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    true_classes = np.array([[5], [6], [7]], dtype=np.int64)
    num_true = 1
    num_sampled = 2
    unique = False
    range_max = 8
    seed = 123
    seed2 = 456
    name = "uniform_candidate_sampler_2"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    true_classes = np.array([[1, 2, 3]], dtype=np.int64)
    num_true = 3
    num_sampled = 5
    unique = True
    range_max = 15
    seed = 789
    seed2 = 101
    name = "uniform_candidate_sampler_3"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    true_classes = np.array([[10, 11], [12, 13], [14, 15]], dtype=np.int64)
    num_true = 2
    num_sampled = 1
    unique = False
    range_max = 20
    seed = 2
    seed2 = 3
    name = "uniform_candidate_sampler_4"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    true_classes = np.array([[1]], dtype=np.int64)
    num_true = 1
    num_sampled = 10
    unique = True
    range_max = 100
    seed = 5
    seed2 = 6
    name = "uniform_candidate_sampler_5"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    true_classes = np.array([[1, 2, 3, 4]], dtype=np.int64)
    num_true = 4
    num_sampled = 7
    unique = False
    range_max = 25
    seed = 10
    seed2 = 20
    name = "uniform_candidate_sampler_6"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    true_classes = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64)
    num_true = 2
    num_sampled = 3
    unique = True
    range_max = 9
    seed = 1
    seed2 = 2
    name = "uniform_candidate_sampler_7"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    true_classes = np.array([[1, 2]], dtype=np.int64)
    num_true = 2
    num_sampled = 2
    unique = False
    range_max = 3
    seed = 3
    seed2 = 4
    name = "uniform_candidate_sampler_8"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    true_classes = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    num_true = 5
    num_sampled = 6
    unique = True
    range_max = 11
    seed = 4
    seed2 = 5
    name = "uniform_candidate_sampler_9"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    true_classes = np.array([[1], [2]], dtype=np.int64)
    num_true = 1
    num_sampled = 3
    unique = False
    range_max = 5
    seed = 6
    seed2 = 7
    name = "uniform_candidate_sampler_10"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    true_classes = np.array([[0,1,2,3,4,5,6,7,8,9]], dtype=np.int64)
    num_true = 10
    num_sampled = 5
    unique = True
    range_max = 20
    seed = 8
    seed2 = 9
    name = "uniform_candidate_sampler_11"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "range_max": range_max, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.UniformCandidateSampler"] = tf_raw_ops_UniformCandidateSampler_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.UniformCandidateSampler' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.UniformCandidateSampler'.")

check_valid('tf.raw_ops.UniformCandidateSampler', generated_inputs['tf.raw_ops.UniformCandidateSampler'], lib="tf", suffix=0)
