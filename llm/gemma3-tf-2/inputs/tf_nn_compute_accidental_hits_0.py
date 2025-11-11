
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_compute_accidental_hits_inputs():
    list_of_inputs = []

    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([2, 5, 1, 6], dtype=np.int64)
    num_true = 2
    seed = 42
    name = "test1"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    sampled_candidates = np.array([2, 4, 6, 8, 10], dtype=np.int64)
    num_true = 3
    seed = 123
    name = "test2"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1], [2]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3], dtype=np.int64)
    num_true = 1
    seed = 0
    name = "test3"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([5, 6, 7, 8], dtype=np.int64)
    num_true = 2
    seed = 99
    name = "test4"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, -2], [3, -4]], dtype=np.int64)
    sampled_candidates = np.array([-2, 5, 1, 6], dtype=np.int64)
    num_true = 2
    seed = 10
    name = "test5"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2, 3, 4]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    num_true = 4
    seed = 50
    name = "test6"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([1, 2], dtype=np.int64)
    num_true = 2
    seed = 77
    name = "test7"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([5, 6, 7, 8], dtype=np.int64)
    num_true = 2
    seed = 1
    name = "test8"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2, 3]], dtype=np.int64)
    sampled_candidates = np.array([3, 4, 5], dtype=np.int64)
    num_true = 3
    seed = 33
    name = "test9"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([1, 3], dtype=np.int64)
    num_true = 2
    seed = 88
    name = "test10"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.compute_accidental_hits"] = tf_nn_compute_accidental_hits_inputs()

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
