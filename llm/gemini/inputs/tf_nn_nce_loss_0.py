
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_nce_loss_inputs():
    list_of_inputs = []

    tf.random.set_seed(1)

    # Input 1
    weights = np.random.rand(10, 5).astype(np.float32)
    biases = np.random.rand(10).astype(np.float32)
    labels = np.random.randint(0, 10, size=(4, 1), dtype=np.int64)
    inputs = np.random.rand(4, 5).astype(np.float32)
    num_sampled = 3
    num_classes = 10
    num_true = 1
    sampled_candidates = np.array([1, 2, 3], dtype=np.int64)
    true_expected_count = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    sampled_expected_count = np.array([0.3, 0.3, 0.4], dtype=np.float32)
    sampled_values = (sampled_candidates, true_expected_count, sampled_expected_count)
    remove_accidental_hits = False
    name = "nce_loss_1"

    input_dict = {
        "weights": weights,
        "biases": biases,
        "labels": labels,
        "inputs": inputs,
        "num_sampled": num_sampled,
        "num_classes": num_classes,
        "num_true": num_true,
        "sampled_values": sampled_values,
        "remove_accidental_hits": remove_accidental_hits,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    weights = np.random.rand(15, 8).astype(np.float32)
    biases = np.random.rand(15).astype(np.float32)
    labels = np.random.randint(0, 15, size=(8, 1), dtype=np.int64)
    inputs = np.random.rand(8, 8).astype(np.float32)
    num_sampled = 5
    num_classes = 15
    num_true = 1
    sampled_candidates = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    true_expected_count = np.array([0.1, 0.1, 0.2, 0.3, 0.3], dtype=np.float32)
    sampled_expected_count = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    sampled_values = (sampled_candidates, true_expected_count, sampled_expected_count)
    remove_accidental_hits = True
    name = "nce_loss_2"

    input_dict = {
        "weights": weights,
        "biases": biases,
        "labels": labels,
        "inputs": inputs,
        "num_sampled": num_sampled,
        "num_classes": num_classes,
        "num_true": num_true,
        "sampled_values": sampled_values,
        "remove_accidental_hits": remove_accidental_hits,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    weights = np.random.rand(20, 10).astype(np.float32)
    biases = np.random.rand(20).astype(np.float32)
    labels = np.random.randint(0, 20, size=(16, 1), dtype=np.int64)
    inputs = np.random.rand(16, 10).astype(np.float32)
    num_sampled = 7
    num_classes = 20
    num_true = 1
    sampled_candidates = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    true_expected_count = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2], dtype=np.float32)
    sampled_expected_count = np.array([0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15], dtype=np.float32)
    sampled_values = (sampled_candidates, true_expected_count, sampled_expected_count)
    remove_accidental_hits = False
    name = "nce_loss_3"

    input_dict = {
        "weights": weights,
        "biases": biases,
        "labels": labels,
        "inputs": inputs,
        "num_sampled": num_sampled,
        "num_classes": num_classes,
        "num_true": num_true,
        "sampled_values": sampled_values,
        "remove_accidental_hits": remove_accidental_hits,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    weights = np.random.rand(5, 3).astype(np.float32)
    biases = np.random.rand(5).astype(np.float32)
    labels = np.random.randint(0, 5, size=(2, 1), dtype=np.int64)
    inputs = np.random.rand(2, 3).astype(np.float32)
    num_sampled = 2
    num_classes = 5
    num_true = 1
    sampled_candidates = np.array([1, 2], dtype=np.int64)
    true_expected_count = np.array([0.6, 0.4], dtype=np.float32)
    sampled_expected_count = np.array([0.5, 0.5], dtype=np.float32)
    sampled_values = (sampled_candidates, true_expected_count, sampled_expected_count)
    remove_accidental_hits = True
    name = "nce_loss_4"

    input_dict = {
        "weights": weights,
        "biases": biases,
        "labels": labels,
        "inputs": inputs,
        "num_sampled": num_sampled,
        "num_classes": num_classes,
        "num_true": num_true,
        "sampled_values": sampled_values,
        "remove_accidental_hits": remove_accidental_hits,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    weights = np.random.rand(12, 6).astype(np.float32)
    biases = np.random.rand(12).astype(np.float32)
    labels = np.random.randint(0, 12, size=(6, 1), dtype=np.int64)
    inputs = np.random.rand(6, 6).astype(np.float32)
    num_sampled = 4
    num_classes = 12
    num_true = 1
    sampled_candidates = np.array([1, 2, 3, 4], dtype=np.int64)
    true_expected_count = np.array([0.2, 0.2, 0.3, 0.3], dtype=np.float32)
    sampled_expected_count = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    sampled_values = (sampled_candidates, true_expected_count, sampled_expected_count)
    remove_accidental_hits = False
    name = "nce_loss_5"

    input_dict = {
        "weights": weights,
        "biases": biases,
        "labels": labels,
        "inputs": inputs,
        "num_sampled": num_sampled,
        "num_classes": num_classes,
        "num_true": num_true,
        "sampled_values": sampled_values,
        "remove_accidental_hits": remove_accidental_hits,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    weights = np.random.rand(8, 4).astype(np.float32)
    biases = np.random.rand(8).astype(np.float32)
    labels = np.random.randint(0, 8, size=(3, 1), dtype=np.int64)
    inputs = np.random.rand(3, 4).astype(np.float32)
    num_sampled = 1
    num_classes = 8
    num_true = 1
    sampled_candidates = np.array([1], dtype=np.int64)
    true_expected_count = np.array([1.0], dtype=np.float32)
    sampled_expected_count = np.array([1.0], dtype=np.float32)
    sampled_values = (sampled_candidates, true_expected_count, sampled_expected_count)
    remove_accidental_hits = True
    name = "nce_loss_6"

    input_dict = {
        "weights": weights,
        "biases": biases,
        "labels": labels,
        "inputs": inputs,
        "num_sampled": num_sampled,
        "num_classes": num_classes,
        "num_true": num_true,
        "sampled_values": sampled_values,
        "remove_accidental_hits": remove_accidental_hits,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    sampled_candidates = np.array([1, 3, 5], dtype=np.int64)
    true_expected_count = np.array([0.5, 0.2, 0.3], dtype=np.float32)
    sampled_expected_count = np.array([0.4, 0.3, 0.3], dtype=np.float32)
    sampled_values = (sampled_candidates, true_expected_count, sampled_expected_count)

    weights = np.random.rand(7, 4).astype(np.float32)
    biases = np.random.rand(7).astype(np.float32)
    labels = np.random.randint(0, 7, size=(3, 1), dtype=np.int64)
    inputs = np.random.rand(3, 4).astype(np.float32)
    num_sampled = 3
    num_classes = 7
    num_true = 1
    remove_accidental_hits = True
    name = "nce_loss_7"

    input_dict = {
        "weights": weights,
        "biases": biases,
        "labels": labels,
        "inputs": inputs,
        "num_sampled": num_sampled,
        "num_classes": num_classes,
        "num_true": num_true,
        "sampled_values": sampled_values,
        "remove_accidental_hits": remove_accidental_hits,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    weights = np.random.rand(9, 5).astype(np.float32)
    biases = np.random.rand(9).astype(np.float32)
    labels = np.random.randint(0, 9, size=(4, 1), dtype=np.int64)
    inputs = np.random.rand(4, 5).astype(np.float32)
    num_sampled = 2
    num_classes = 9
    num_true = 1
    sampled_candidates = np.array([1, 2], dtype=np.int64)
    true_expected_count = np.array([0.5, 0.5], dtype=np.float32)
    sampled_expected_count = np.array([0.5, 0.5], dtype=np.float32)

    sampled_values = (sampled_candidates, true_expected_count, sampled_expected_count)
    remove_accidental_hits = False
    name = "nce_loss_8"

    input_dict = {
        "weights": weights,
        "biases": biases,
        "labels": labels,
        "inputs": inputs,
        "num_sampled": num_sampled,
        "num_classes": num_classes,
        "num_true": num_true,
        "sampled_values": sampled_values,
        "remove_accidental_hits": remove_accidental_hits,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    weights = np.random.rand(11, 7).astype(np.float32)
    biases = np.random.rand(11).astype(np.float32)
    labels = np.random.randint(0, 11, size=(5, 1), dtype=np.int64)
    inputs = np.random.rand(5, 7).astype(np.float32)
    num_sampled = 6
    num_classes = 11
    num_true = 1

    sampled_candidates = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    true_expected_count = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6], dtype=np.float32)
    sampled_expected_count = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6], dtype=np.float32)

    sampled_values = (sampled_candidates, true_expected_count, sampled_expected_count)
    remove_accidental_hits = True
    name = "nce_loss_9"

    input_dict = {
        "weights": weights,
        "biases": biases,
        "labels": labels,
        "inputs": inputs,
        "num_sampled": num_sampled,
        "num_classes": num_classes,
        "num_true": num_true,
        "sampled_values": sampled_values,
        "remove_accidental_hits": remove_accidental_hits,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    weights = np.random.rand(13, 9).astype(np.float32)
    biases = np.random.rand(13).astype(np.float32)
    labels = np.random.randint(0, 13, size=(7, 1), dtype=np.int64)
    inputs = np.random.rand(7, 9).astype(np.float32)
    num_sampled = 8
    num_classes = 13
    num_true = 1

    sampled_candidates = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    true_expected_count = np.array([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8], dtype=np.float32)
    sampled_expected_count = np.array([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8], dtype=np.float32)

    sampled_values = (sampled_candidates, true_expected_count, sampled_expected_count)
    remove_accidental_hits = False
    name = "nce_loss_10"

    input_dict = {
        "weights": weights,
        "biases": biases,
        "labels": labels,
        "inputs": inputs,
        "num_sampled": num_sampled,
        "num_classes": num_classes,
        "num_true": num_true,
        "sampled_values": sampled_values,
        "remove_accidental_hits": remove_accidental_hits,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.nce_loss"] = tf_nn_nce_loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.nce_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.nce_loss'.")

check_valid('tf.nn.nce_loss', generated_inputs['tf.nn.nce_loss'], lib="tf", suffix=0)
