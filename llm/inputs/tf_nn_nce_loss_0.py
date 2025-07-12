
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_nce_loss_inputs():
    list_of_inputs = []

    # Input 1
    weights = np.float32(np.random.rand(1000, 128))
    biases = np.float32(np.random.rand(1000))
    labels = np.int64(np.random.randint(0, 1000, size=(32, 1)))
    inputs = np.float32(np.random.rand(32, 128))
    num_sampled = 25
    num_classes = 1000
    num_true = 1
    sampled_values = None
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
    weights = np.float32(np.random.rand(500, 64))
    biases = np.float32(np.random.rand(500))
    labels = np.int64(np.random.randint(0, 500, size=(64, 1)))
    inputs = np.float32(np.random.rand(64, 64))
    num_sampled = 10
    num_classes = 500
    num_true = 1
    sampled_values = None
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
    weights = np.float32(np.random.rand(2000, 256))
    biases = np.float32(np.random.rand(2000))
    labels = np.int64(np.random.randint(0, 2000, size=(128, 5)))
    inputs = np.float32(np.random.rand(128, 256))
    num_sampled = 50
    num_classes = 2000
    num_true = 5
    sampled_values = None
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
    weights = np.float32(np.random.rand(100, 32))
    biases = np.float32(np.random.rand(100))
    labels = np.int64(np.random.randint(0, 100, size=(16, 2)))
    inputs = np.float32(np.random.rand(16, 32))
    num_sampled = 5
    num_classes = 100
    num_true = 2
    sampled_candidates = np.int64(np.random.randint(0, 100, size=(num_sampled,)))
    true_expected_count = np.float32(np.random.rand(16, 2, num_sampled))
    sampled_expected_count = np.float32(np.random.rand(num_sampled))
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

    # Input 5: different batch size
    weights = np.float32(np.random.rand(1000, 128))
    biases = np.float32(np.random.rand(1000))
    labels = np.int64(np.random.randint(0, 1000, size=(64, 1)))
    inputs = np.float32(np.random.rand(64, 128))
    num_sampled = 25
    num_classes = 1000
    num_true = 1
    sampled_values = None
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

    # Input 6: num_true > 1 and accidental hits removed
    weights = np.float32(np.random.rand(500, 64))
    biases = np.float32(np.random.rand(500))
    labels = np.int64(np.random.randint(0, 500, size=(32, 3)))
    inputs = np.float32(np.random.rand(32, 64))
    num_sampled = 10
    num_classes = 500
    num_true = 3
    sampled_values = None
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

    # Input 7: small values
    weights = np.float32(np.random.rand(10, 5) * 0.01)
    biases = np.float32(np.random.rand(10) * 0.01)
    labels = np.int64(np.random.randint(0, 10, size=(2, 1)))
    inputs = np.float32(np.random.rand(2, 5) * 0.01)
    num_sampled = 2
    num_classes = 10
    num_true = 1
    sampled_values = None
    remove_accidental_hits = False
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

   # Input 8: Different name
    weights = np.float32(np.random.rand(1000, 128))
    biases = np.float32(np.random.rand(1000))
    labels = np.int64(np.random.randint(0, 1000, size=(32, 1)))
    inputs = np.float32(np.random.rand(32, 128))
    num_sampled = 25
    num_classes = 1000
    num_true = 1
    sampled_values = None
    remove_accidental_hits = False
    name = "another_name"

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
    weights = np.float32(np.random.rand(50, 16))
    biases = np.float32(np.random.rand(50))
    labels = np.int64(np.random.randint(0, 50, size=(8, 1)))
    inputs = np.float32(np.random.rand(8, 16))
    num_sampled = 3
    num_classes = 50
    num_true = 1
    sampled_candidates = np.int64(np.random.randint(0, 50, size=(num_sampled,)))
    true_expected_count = np.float32(np.random.rand(8, 1, num_sampled))
    sampled_expected_count = np.float32(np.random.rand(num_sampled))
    sampled_values = (sampled_candidates, true_expected_count, sampled_expected_count)
    remove_accidental_hits = False
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
