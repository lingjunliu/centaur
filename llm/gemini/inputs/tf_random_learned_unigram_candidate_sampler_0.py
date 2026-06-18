
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_random_learned_unigram_candidate_sampler_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'true_classes': np.array([[1, 2]], dtype=np.int64),
        'num_true': 2,
        'num_sampled': 5,
        'unique': True,
        'range_max': 10,
        'seed': 42,
        'name': "sampler_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'true_classes': np.array([[0], [4], [3]], dtype=np.int64),
        'num_true': 1,
        'num_sampled': 3,
        'unique': False,
        'range_max': 5,
        'seed': 10,
        'name': "sampler_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'true_classes': np.array([[10, 20, 30], [5, 15, 25]], dtype=np.int64),
        'num_true': 3,
        'num_sampled': 10,
        'unique': True,
        'range_max': 100,
        'seed': 123,
        'name': "sampler_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'true_classes': np.array([[0, 1, 2, 3, 4]], dtype=np.int64),
        'num_true': 5,
        'num_sampled': 2,
        'unique': False,
        'range_max': 10,
        'seed': 0,
        'name': "sampler_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'true_classes': np.array([[0], [1]], dtype=np.int64),
        'num_true': 1,
        'num_sampled': 1,
        'unique': True,
        'range_max': 2,
        'seed': 99,
        'name': "sampler_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'true_classes': np.array([[5, 5], [3, 3]], dtype=np.int64),
        'num_true': 2,
        'num_sampled': 4,
        'unique': False,
        'range_max': 6,
        'seed': 7,
        'name': "sampler_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'true_classes': np.array([[99], [50], [0], [25]], dtype=np.int64),
        'num_true': 1,
        'num_sampled': 20,
        'unique': True,
        'range_max': 1000,
        'seed': 777,
        'name': "sampler_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'true_classes': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
        'num_true': 3,
        'num_sampled': 5,
        'unique': True,
        'range_max': 15,
        'seed': 888,
        'name': "sampler_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'true_classes': np.array([[0, 0], [0, 0]], dtype=np.int64),
        'num_true': 2,
        'num_sampled': 1,
        'unique': False,
        'range_max': 1,
        'seed': 1,
        'name': "sampler_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'true_classes': np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64),
        'num_true': 6,
        'num_sampled': 3,
        'unique': True,
        'range_max': 7,
        'seed': 55,
        'name': "sampler_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.learned_unigram_candidate_sampler"] = tf_random_learned_unigram_candidate_sampler_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.learned_unigram_candidate_sampler' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.learned_unigram_candidate_sampler'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.learned_unigram_candidate_sampler', generated_inputs['tf.random.learned_unigram_candidate_sampler'], lib="tf", suffix=0)
