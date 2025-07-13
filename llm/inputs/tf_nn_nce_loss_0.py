
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_nce_loss_inputs():
    list_of_inputs = []

    # Input 1
    weights = np.random.rand(100, 128).astype(np.float32)
    biases = np.random.rand(100).astype(np.float32)
    labels = np.random.randint(0, 100, size=(32, 1), dtype=np.int64)
    inputs = np.random.rand(32, 128).astype(np.float32)
    num_sampled = 25
    num_classes = 100
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
    weights = np.random.rand(50, 64).astype(np.float32)
    biases = np.random.rand(50).astype(np.float32)
    labels = np.random.randint(0, 50, size=(64, 1), dtype=np.int64)
    inputs = np.random.rand(64, 64).astype(np.float32)
    num_sampled = 10
    num_classes = 50
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
    weights = np.random.rand(200, 256).astype(np.float32)
    biases = np.random.rand(200).astype(np.float32)
    labels = np.random.randint(0, 200, size=(128, 1), dtype=np.int64)
    inputs = np.random.rand(128, 256).astype(np.float32)
    num_sampled = 50
    num_classes = 200
    num_true = 1
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
    weights = np.random.rand(100, 128).astype(np.float32)
    biases = np.random.rand(100).astype(np.float32)
    labels = np.random.randint(0, 100, size=(32, 3), dtype=np.int64)
    inputs = np.random.rand(32, 128).astype(np.float32)
    num_sampled = 25
    num_classes = 100
    num_true = 3
    sampled_values = None
    remove_accidental_hits = False
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
    weights = np.random.rand(50, 64).astype(np.float32)
    biases = np.random.rand(50).astype(np.float32)
    labels = np.random.randint(0, 50, size=(64, 2), dtype=np.int64)
    inputs = np.random.rand(64, 64).astype(np.float32)
    num_sampled = 10
    num_classes = 50
    num_true = 2
    sampled_values = None
    remove_accidental_hits = True
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
    weights = np.random.rand(100, 128).astype(np.float32)
    biases = np.random.rand(100).astype(np.float32)
    labels = np.random.randint(0, 100, size=(32, 1), dtype=np.int64)
    inputs = np.random.rand(32, 128).astype(np.float32)
    num_sampled = 25
    num_classes = 100
    num_true = 1
    sampled_values = (np.array([1,2,3], dtype=np.int64), np.array([0.1,0.2,0.3], dtype=np.float32), np.array([0.4,0.5,0.6], dtype=np.float32))
    remove_accidental_hits = False
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
    weights = np.random.rand(100, 128).astype(np.float32)
    biases = np.random.rand(100).astype(np.float32)
    labels = np.random.randint(0, 100, size=(32, 1), dtype=np.int64)
    inputs = np.random.rand(32, 128).astype(np.float32)
    num_sampled = 25
    num_classes = 100
    num_true = 1
    sampled_values = (np.array([1,2,3], dtype=np.int64), np.array([0.1,0.2,0.3], dtype=np.float32), np.array([0.4,0.5,0.6], dtype=np.float32))
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

    # Input 8
    weights = np.random.rand(50, 64).astype(np.float32)
    biases = np.random.rand(50).astype(np.float32)
    labels = np.random.randint(0, 50, size=(64, 2), dtype=np.int64)
    inputs = np.random.rand(64, 64).astype(np.float32)
    num_sampled = 10
    num_classes = 50
    num_true = 2
    sampled_values = (np.array([4,5,6], dtype=np.int64), np.array([0.7,0.8,0.9], dtype=np.float32), np.array([1.0,1.1,1.2], dtype=np.float32))
    remove_accidental_hits = True
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
    weights = np.random.rand(200, 256).astype(np.float32)
    biases = np.random.rand(200).astype(np.float32)
    labels = np.random.randint(0, 200, size=(128, 4), dtype=np.int64)
    inputs = np.random.rand(128, 256).astype(np.float32)
    num_sampled = 50
    num_classes = 200
    num_true = 4
    sampled_values = (np.array([7,8,9], dtype=np.int64), np.array([1.3,1.4,1.5], dtype=np.float32), np.array([1.6,1.7,1.8], dtype=np.float32))
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
