
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_sampled_softmax_loss_inputs():
    list_of_inputs = []

    def generate_sampled_values(num_sampled, seed=None):
        if seed is not None:
            np.random.seed(seed)
        sampled_candidates = np.random.randint(0, 100, size=(num_sampled,), dtype=np.int64)
        true_expected_count = np.random.rand(num_sampled).astype(np.float32)
        sampled_expected_count = np.random.rand(num_sampled).astype(np.float32)
        return (sampled_candidates, true_expected_count, sampled_expected_count)

    # Input 1
    weights = np.random.rand(100, 128).astype(np.float32)
    biases = np.random.rand(100).astype(np.float32)
    labels = np.random.randint(0, 100, size=(32, 1), dtype=np.int64)
    inputs = np.random.rand(32, 128).astype(np.float32)
    num_sampled = 25
    num_classes = 100
    num_true = 1
    sampled_values = generate_sampled_values(num_sampled, seed=123)
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
    num_sampled = 100
    num_classes = 500
    num_true = 1
    sampled_values = generate_sampled_values(num_sampled, seed=456)
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
    weights = np.random.rand(200, 32).astype(np.float32)
    biases = np.random.rand(200).astype(np.float32)
    labels = np.random.randint(0, 200, size=(16, 1), dtype=np.int64)
    inputs = np.random.rand(16, 32).astype(np.float32)
    num_sampled = 50
    num_classes = 200
    num_true = 1
    sampled_values = generate_sampled_values(num_sampled)
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
    
    # Input 4
    weights = np.random.rand(1000, 128).astype(np.float32)
    biases = np.random.rand(1000).astype(np.float32)
    labels = np.random.randint(0, 1000, size=(128, 1), dtype=np.int64)
    inputs = np.random.rand(128, 128).astype(np.float32)
    num_sampled = 200
    num_classes = 1000
    num_true = 1
    sampled_values = generate_sampled_values(num_sampled, seed=789)
    remove_accidental_hits = False
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
    
    # Input 5
    weights = np.random.rand(50, 16).astype(np.float32)
    biases = np.random.rand(50).astype(np.float32)
    labels = np.random.randint(0, 50, size=(8, 1), dtype=np.int64)
    inputs = np.random.rand(8, 16).astype(np.float32)
    num_sampled = 10
    num_classes = 50
    num_true = 1
    sampled_values = generate_sampled_values(num_sampled, seed=101)
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

    # Input 6: Multiple true labels
    weights = np.random.rand(100, 128).astype(np.float32)
    biases = np.random.rand(100).astype(np.float32)
    labels = np.random.randint(0, 100, size=(32, 3), dtype=np.int64)
    inputs = np.random.rand(32, 128).astype(np.float32)
    num_sampled = 25
    num_classes = 100
    num_true = 3
    sampled_values = generate_sampled_values(num_sampled, seed=123)
    remove_accidental_hits = True
    seed = 123
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

    # Input 7: Small number of sampled
    weights = np.random.rand(100, 128).astype(np.float32)
    biases = np.random.rand(100).astype(np.float32)
    labels = np.random.randint(0, 100, size=(32, 1), dtype=np.int64)
    inputs = np.random.rand(32, 128).astype(np.float32)
    num_sampled = 5
    num_classes = 100
    num_true = 1
    sampled_values = generate_sampled_values(num_sampled, seed=123)
    remove_accidental_hits = True
    seed = 123
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

    # Input 8: Large batch size
    weights = np.random.rand(100, 128).astype(np.float32)
    biases = np.random.rand(100).astype(np.float32)
    labels = np.random.randint(0, 100, size=(256, 1), dtype=np.int64)
    inputs = np.random.rand(256, 128).astype(np.float32)
    num_sampled = 25
    num_classes = 100
    num_true = 1
    sampled_values = generate_sampled_values(num_sampled, seed=123)
    remove_accidental_hits = True
    seed = 123
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
    
    # Input 9: sampled_values provided
    sampled_candidates = np.array([1, 2, 3], dtype=np.int64)
    true_expected_count = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    sampled_expected_count = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    sampled_values = (sampled_candidates, true_expected_count, sampled_expected_count)
    
    weights = np.random.rand(100, 128).astype(np.float32)
    biases = np.random.rand(100).astype(np.float32)
    labels = np.random.randint(0, 100, size=(32, 1), dtype=np.int64)
    inputs = np.random.rand(32, 128).astype(np.float32)
    num_sampled = 3
    num_classes = 100
    num_true = 1
    remove_accidental_hits = True
    seed = 123
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

     # Input 10: Different dimensions of inputs and weights
    weights = np.random.rand(100, 64).astype(np.float32)
    biases = np.random.rand(100).astype(np.float32)
    labels = np.random.randint(0, 100, size=(32, 1), dtype=np.int64)
    inputs = np.random.rand(32, 64).astype(np.float32)
    num_sampled = 25
    num_classes = 100
    num_true = 1
    sampled_values = generate_sampled_values(num_sampled, seed=123)
    remove_accidental_hits = True
    seed = 123
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
