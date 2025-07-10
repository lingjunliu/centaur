
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_sampled_softmax_loss_inputs():
    list_of_inputs = []

    # Helper function to create a tuple of sampled values
    def create_sampled_values(num_sampled, num_classes, batch_size, num_true):
        sampled_candidates = tf.constant(np.random.randint(0, num_classes, size=(num_sampled,)), dtype=tf.int64).numpy()
        true_expected_count = tf.constant(np.random.rand(batch_size, num_true, 1), dtype=tf.float32).numpy()
        sampled_expected_count = tf.constant(np.random.rand(num_sampled), dtype=tf.float32).numpy()
        return (sampled_candidates, true_expected_count, sampled_expected_count)

    # Input 1
    weights = tf.constant(np.random.rand(100, 128), dtype=tf.float32).numpy()
    biases = tf.constant(np.random.rand(100), dtype=tf.float32).numpy()
    labels = tf.constant(np.random.randint(0, 100, size=(32, 1)), dtype=tf.int64).numpy()
    inputs = tf.constant(np.random.rand(32, 128), dtype=tf.float32).numpy()
    num_sampled = 25
    num_classes = 100
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
    weights = tf.constant(np.random.rand(50, 64), dtype=tf.float32).numpy()
    biases = tf.constant(np.random.rand(50), dtype=tf.float32).numpy()
    labels = tf.constant(np.random.randint(0, 50, size=(16, 1)), dtype=tf.int64).numpy()
    inputs = tf.constant(np.random.rand(16, 64), dtype=tf.float32).numpy()
    num_sampled = 10
    num_classes = 50
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
    weights = tf.constant(np.random.rand(200, 256), dtype=tf.float32).numpy()
    biases = tf.constant(np.random.rand(200), dtype=tf.float32).numpy()
    labels = tf.constant(np.random.randint(0, 200, size=(64, 1)), dtype=tf.int64).numpy()
    inputs = tf.constant(np.random.rand(64, 256), dtype=tf.float32).numpy()
    num_sampled = 50
    num_classes = 200
    num_true = 1
    sampled_values = create_sampled_values(num_sampled, num_classes, 64, num_true)
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
    weights = tf.constant(np.random.rand(80, 32), dtype=tf.float32).numpy()
    biases = tf.constant(np.random.rand(80), dtype=tf.float32).numpy()
    labels = tf.constant(np.random.randint(0, 80, size=(8, 1)), dtype=tf.int64).numpy()
    inputs = tf.constant(np.random.rand(8, 32), dtype=tf.float32).numpy()
    num_sampled = 5
    num_classes = 80
    num_true = 1
    sampled_values = create_sampled_values(num_sampled, num_classes, 8, num_true)
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
    weights = tf.constant(np.random.rand(150, 100), dtype=tf.float32).numpy()
    biases = tf.constant(np.random.rand(150), dtype=tf.float32).numpy()
    labels = tf.constant(np.random.randint(0, 150, size=(48, 1)), dtype=tf.int64).numpy()
    inputs = tf.constant(np.random.rand(48, 100), dtype=tf.float32).numpy()
    num_sampled = 30
    num_classes = 150
    num_true = 1
    sampled_values = create_sampled_values(num_sampled, num_classes, 48, num_true)
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
    
    # Input 6: num_true = 2
    weights = tf.constant(np.random.rand(120, 80), dtype=tf.float32).numpy()
    biases = tf.constant(np.random.rand(120), dtype=tf.float32).numpy()
    labels = tf.constant(np.random.randint(0, 120, size=(24, 2)), dtype=tf.int64).numpy()
    inputs = tf.constant(np.random.rand(24, 80), dtype=tf.float32).numpy()
    num_sampled = 20
    num_classes = 120
    num_true = 2
    sampled_values = create_sampled_values(num_sampled, num_classes, 24, num_true)
    remove_accidental_hits = False
    seed = 222
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

    # Input 7: with sampled_values
    weights = tf.constant(np.random.rand(70, 48), dtype=tf.float32).numpy()
    biases = tf.constant(np.random.rand(70), dtype=tf.float32).numpy()
    labels = tf.constant(np.random.randint(0, 70, size=(12, 1)), dtype=tf.int64).numpy()
    inputs = tf.constant(np.random.rand(12, 48), dtype=tf.float32).numpy()
    num_sampled = 15
    num_classes = 70
    num_true = 1
    sampled_values = create_sampled_values(num_sampled, num_classes, 12, num_true)
    remove_accidental_hits = True
    seed = 333
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

    # Input 8: Different batch size and dimensions
    weights = tf.constant(np.random.rand(90, 72), dtype=tf.float32).numpy()
    biases = tf.constant(np.random.rand(90), dtype=tf.float32).numpy()
    labels = tf.constant(np.random.randint(0, 90, size=(36, 1)), dtype=tf.int64).numpy()
    inputs = tf.constant(np.random.rand(36, 72), dtype=tf.float32).numpy()
    num_sampled = 18
    num_classes = 90
    num_true = 1
    sampled_values = create_sampled_values(num_sampled, num_classes, 36, num_true)
    remove_accidental_hits = False
    seed = 444
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
    
    # Input 9: smaller values
    weights = tf.constant(np.random.rand(30, 16) * 0.1, dtype=tf.float32).numpy()
    biases = tf.constant(np.random.rand(30) * 0.1, dtype=tf.float32).numpy()
    labels = tf.constant(np.random.randint(0, 30, size=(6, 1)), dtype=tf.int64).numpy()
    inputs = tf.constant(np.random.rand(6, 16) * 0.1, dtype=tf.float32).numpy()
    num_sampled = 6
    num_classes = 30
    num_true = 1
    sampled_values = create_sampled_values(num_sampled, num_classes, 6, num_true)
    remove_accidental_hits = True
    seed = 555
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

    # Input 10: Larger Values
    weights = tf.constant(np.random.rand(40, 24) * 10, dtype=tf.float32).numpy()
    biases = tf.constant(np.random.rand(40) * 10, dtype=tf.float32).numpy()
    labels = tf.constant(np.random.randint(0, 40, size=(10, 1)), dtype=tf.int64).numpy()
    inputs = tf.constant(np.random.rand(10, 24) * 10, dtype=tf.float32).numpy()
    num_sampled = 8
    num_classes = 40
    num_true = 1
    sampled_values = create_sampled_values(num_sampled, num_classes, 10, num_true)
    remove_accidental_hits = False
    seed = 666
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
