
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_all_candidate_sampler_inputs():
    list_of_inputs = []

    # Input 1
    true_classes = np.array([[1, 2]], dtype=np.int64)
    num_true = 2
    num_sampled = 5
    unique = True
    seed = 123
    name = "sampler_1"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    true_classes = np.array([[0, 1], [2, 3]], dtype=np.int64)
    num_true = 2
    num_sampled = 10
    unique = False
    seed = 456
    name = None
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    true_classes = np.array([[5]], dtype=np.int64)
    num_true = 1
    num_sampled = 8
    unique = True
    seed = 42
    name = "sampler_3"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    true_classes = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    num_true = 3
    num_sampled = 7
    unique = False
    seed = 789
    name = "sampler_4"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    true_classes = np.array([[0]], dtype=np.int64)
    num_true = 1
    num_sampled = 1
    unique = True
    seed = 1
    name = "sampler_5"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    true_classes = np.array([[1, 2, 3, 4]], dtype=np.int64)
    num_true = 4
    num_sampled = 15
    unique = False
    seed = 100
    name = "sampler_6"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    num_true = 2
    num_sampled = 5
    unique = True
    seed = 111
    name = None
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8. Making num_sampled larger than true_classes values
    true_classes = np.array([[0]], dtype=np.int64)
    num_true = 1
    num_sampled = 2000
    unique = False
    seed = 666
    name = "all_candidate_sampler_8"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9. Keeping true_classes values within the range [0, num_sampled)
    true_classes = np.array([[0, 1]], dtype=np.int64)
    num_true = 2
    num_sampled = 5
    unique = True
    seed = 99
    name = "sampler_9"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    true_classes = np.array([[1]], dtype=np.int64)
    num_true = 1
    num_sampled = 100
    unique = False
    seed = 10
    name = "sampler_10"
    input_dict = {"true_classes": true_classes, "num_true": num_true, "num_sampled": num_sampled, "unique": unique, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.all_candidate_sampler"] = tf_random_all_candidate_sampler_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.all_candidate_sampler' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.all_candidate_sampler'.")

check_valid('tf.random.all_candidate_sampler', generated_inputs['tf.random.all_candidate_sampler'], lib="tf", suffix=0)
