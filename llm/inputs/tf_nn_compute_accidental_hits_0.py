
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_compute_accidental_hits_inputs():
    list_of_inputs = []

    # Input 1
    true_classes = np.array([[1]], dtype=np.int64)
    sampled_candidates = np.array([1, 3, 5, 2], dtype=np.int64)
    num_true = 1
    seed = 123
    name = "test1"
    input_dict = {
        "true_classes": tf.constant(true_classes),
        "sampled_candidates": tf.constant(sampled_candidates),
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    true_classes = np.array([[1, 2]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    num_true = 2
    seed = 1
    name = "test2"
    input_dict = {
        "true_classes": tf.constant(true_classes),
        "sampled_candidates": tf.constant(sampled_candidates),
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    true_classes = np.array([[1]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3], dtype=np.int64)
    num_true = 1
    seed = 42
    name = "test3"
    input_dict = {
        "true_classes": tf.constant(true_classes),
        "sampled_candidates": tf.constant(sampled_candidates),
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    true_classes = np.array([[1, 2]], dtype=np.int64)
    sampled_candidates = np.array([3, 4, 5, 6], dtype=np.int64)
    num_true = 2
    seed = 0
    name = "test4"
    input_dict = {
        "true_classes": tf.constant(true_classes),
        "sampled_candidates": tf.constant(sampled_candidates),
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    true_classes = np.array([[10, 20], [30, 40]], dtype=np.int64)
    sampled_candidates = np.array([10, 30, 50, 60], dtype=np.int64)
    num_true = 2
    seed = 1
    name = "test5"
    input_dict = {
        "true_classes": tf.constant(true_classes),
        "sampled_candidates": tf.constant(sampled_candidates),
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    true_classes = np.array([[1, 2, 3, 4]], dtype=np.int64)
    sampled_candidates = np.array([2, 5, 7, 4], dtype=np.int64)
    num_true = 4
    seed = 2
    name = "test6"
    input_dict = {
        "true_classes": tf.constant(true_classes),
        "sampled_candidates": tf.constant(sampled_candidates),
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    true_classes = np.array([[1], [2]], dtype=np.int64)
    sampled_candidates = np.array([1, 2], dtype=np.int64)
    num_true = 1
    seed = 3
    name = "test7"
    input_dict = {
        "true_classes": tf.constant(true_classes),
        "sampled_candidates": tf.constant(sampled_candidates),
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4], dtype=np.int64)
    num_true = 2
    seed = 4
    name = "test8"
    input_dict = {
        "true_classes": tf.constant(true_classes),
        "sampled_candidates": tf.constant(sampled_candidates),
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    true_classes = np.array([[1, 5], [2, 6], [3, 7]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3], dtype=np.int64)
    num_true = 2
    seed = 5
    name = "test9"
    input_dict = {
        "true_classes": tf.constant(true_classes),
        "sampled_candidates": tf.constant(sampled_candidates),
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    true_classes = np.array([[1, 2]], dtype=np.int64)
    sampled_candidates = np.array([3, 4, 5, 6], dtype=np.int64)
    num_true = 2
    seed = 6
    name = "test10"
    input_dict = {
        "true_classes": tf.constant(true_classes),
        "sampled_candidates": tf.constant(sampled_candidates),
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.compute_accidental_hits"] = tf_nn_compute_accidental_hits_inputs()
for i in range(len(generated_inputs["tf.nn.compute_accidental_hits"])):
    generated_inputs["tf.nn.compute_accidental_hits"][i]["true_classes"] = generated_inputs["tf.nn.compute_accidental_hits"][i]["true_classes"].numpy()
    generated_inputs["tf.nn.compute_accidental_hits"][i]["sampled_candidates"] = generated_inputs["tf.nn.compute_accidental_hits"][i]["sampled_candidates"].numpy()
    tf.random.set_seed(generated_inputs["tf.nn.compute_accidental_hits"][i]["seed"] if generated_inputs["tf.nn.compute_accidental_hits"][i]["seed"] is not None else 0)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.compute_accidental_hits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.compute_accidental_hits'.")

check_valid('tf.nn.compute_accidental_hits', generated_inputs['tf.nn.compute_accidental_hits'], lib="tf", suffix=0)
