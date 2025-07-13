
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_sampled_softmax_loss_inputs():
    list_of_inputs = []

    # Input 1
    weights = np.random.rand(1000, 128).astype(np.float32)
    biases = np.random.rand(1000).astype(np.float32)
    labels = np.random.randint(0, 1000, size=(32, 1), dtype=np.int64)
    inputs = np.random.rand(32, 128).astype(np.float32)
    num_sampled = 25
    num_classes = 1000
    num_true = 1
    sampled_values = None
    remove_accidental_hits = True
    seed = 123
    name = "loss1"

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
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    weights = np.random.rand(500, 64).astype(np.float32)
    biases = np.random.rand(500).astype(np.float32)
    labels = np.random.randint(0, 500, size=(64, 1), dtype=np.int64)
    inputs = np.random.rand(64, 64).astype(np.float32)
    num_sampled = 10
    num_classes = 500
    num_true = 1
    sampled_values = None
    remove_accidental_hits = False
    seed = 456
    name = "loss2"

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
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    weights = np.random.rand(2000, 256).astype(np.float32)
    biases = np.random.rand(2000).astype(np.float32)
    labels = np.random.randint(0, 2000, size=(128, 1), dtype=np.int64)
    inputs = np.random.rand(128, 256).astype(np.float32)
    num_sampled = 50
    num_classes = 2000
    num_true = 1
    sampled_values = None
    remove_accidental_hits = True
    seed = None
    name = "loss3"

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
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4: Multiple true labels per example
    weights = np.random.rand(100, 32).astype(np.float32)
    biases = np.random.rand(100).astype(np.float32)
    labels = np.random.randint(0, 100, size=(16, 3), dtype=np.int64)
    inputs = np.random.rand(16, 32).astype(np.float32)
    num_sampled = 10
    num_classes = 100
    num_true = 3
    sampled_values = None
    remove_accidental_hits = True
    seed = 789
    name = "loss4"

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
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small number of classes
    weights = np.random.rand(10, 8).astype(np.float32)
    biases = np.random.rand(10).astype(np.float32)
    labels = np.random.randint(0, 10, size=(8, 1), dtype=np.int64)
    inputs = np.random.rand(8, 8).astype(np.float32)
    num_sampled = 5
    num_classes = 10
    num_true = 1
    sampled_values = None
    remove_accidental_hits = True
    seed = 101
    name = "loss5"

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
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  sampled_values specified
    sampled_candidates = np.array([1, 5, 2, 8], dtype=np.int64)
    true_expected_count = np.array([0.8, 0.2, 0.9, 0.1], dtype=np.float32)
    sampled_expected_count = np.array([0.2, 0.8, 0.1, 0.9], dtype=np.float32)

    weights = np.random.rand(20, 4).astype(np.float32)
    biases = np.random.rand(20).astype(np.float32)
    labels = np.random.randint(0, 20, size=(4, 1), dtype=np.int64)
    inputs = np.random.rand(4, 4).astype(np.float32)
    num_sampled = 4
    num_classes = 20
    num_true = 1
    sampled_values = (sampled_candidates, true_expected_count, sampled_expected_count)
    remove_accidental_hits = True
    seed = None
    name = "loss6"

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
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: num_sampled equal to num_classes, setting sampled_values to None
    weights = np.random.rand(50, 16).astype(np.float32)
    biases = np.random.rand(50).astype(np.float32)
    labels = np.random.randint(0, 50, size=(16, 1), dtype=np.int64)
    inputs = np.random.rand(16, 16).astype(np.float32)
    num_sampled = 50
    num_classes = 50
    num_true = 1
    sampled_values = None
    remove_accidental_hits = True
    seed = 112
    name = "loss7"

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
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different batch size
    weights = np.random.rand(300, 32).astype(np.float32)
    biases = np.random.rand(300).astype(np.float32)
    labels = np.random.randint(0, 300, size=(64, 1), dtype=np.int64)
    inputs = np.random.rand(64, 32).astype(np.float32)
    num_sampled = 20
    num_classes = 300
    num_true = 1
    sampled_values = None
    remove_accidental_hits = False
    seed = 223
    name = "loss8"

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
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Change the input dimensions
    weights = np.random.rand(400, 64).astype(np.float32)
    biases = np.random.rand(400).astype(np.float32)
    labels = np.random.randint(0, 400, size=(32, 1), dtype=np.int64)
    inputs = np.random.rand(32, 64).astype(np.float32)
    num_sampled = 15
    num_classes = 400
    num_true = 1
    sampled_values = None
    remove_accidental_hits = True
    seed = 334
    name = "loss9"

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
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different Seed and more samples
    weights = np.random.rand(600, 128).astype(np.float32)
    biases = np.random.rand(600).astype(np.float32)
    labels = np.random.randint(0, 600, size=(128, 1), dtype=np.int64)
    inputs = np.random.rand(128, 128).astype(np.float32)
    num_sampled = 75
    num_classes = 600
    num_true = 1
    sampled_values = None
    remove_accidental_hits = True
    seed = 445
    name = "loss10"

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
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.sampled_softmax_loss"] = tf_nn_sampled_softmax_loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.sampled_softmax_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.sampled_softmax_loss'.")

check_valid('tf.nn.sampled_softmax_loss', generated_inputs['tf.nn.sampled_softmax_loss'], lib="tf", suffix=0)
