
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def compute_accidental_hits_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    true_classes = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    sampled_candidates = np.array([2, 4, 5, 6, 7], dtype=np.int64)
    num_true = 3
    seed = 0
    name = "accidental_hit_1"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    true_classes = np.array([[1], [4], [5]], dtype=np.int64)
    sampled_candidates = np.array([1, 4, 5, 7], dtype=np.int64)
    num_true = 1
    seed = 42
    name = "accidental_hit_2"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    true_classes = np.array([[10, 20], [30, 40]], dtype=np.int64)
    sampled_candidates = np.array([10, 20, 30, 40], dtype=np.int64)
    num_true = 2
    seed = 123
    name = "accidental_hit_3"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    true_classes = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    num_true = 5
    seed = 456
    name = "accidental_hit_4"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    true_classes = np.array([[100], [200]], dtype=np.int64)
    sampled_candidates = np.array([100, 200, 300], dtype=np.int64)
    num_true = 1
    seed = 789
    name = "accidental_hit_5"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    true_classes = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    num_true = 2
    seed = 0
    name = "accidental_hit_6"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    true_classes = np.array([[1], [2], [3], [4]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    num_true = 1
    seed = 10
    name = "accidental_hit_7"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    true_classes = np.array([[100, 200, 300], [400, 500, 600]], dtype=np.int64)
    sampled_candidates = np.array([100, 200, 300, 400, 500], dtype=np.int64)
    num_true = 3
    seed = 20
    name = "accidental_hit_8"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    true_classes = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    num_true = 4
    seed = 30
    name = "accidental_hit_9"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    true_classes = np.array([[1], [2], [3]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    num_true = 1
    seed = 40
    name = "accidental_hit_10"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.compute_accidental_hits"] = compute_accidental_hits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.compute_accidental_hits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.compute_accidental_hits'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.compute_accidental_hits', generated_inputs['tf.nn.compute_accidental_hits'], lib="tf", suffix=0)
