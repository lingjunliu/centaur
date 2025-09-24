
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_compute_average_loss_inputs():
    list_of_inputs = []

    # Input 1: Basic case with no sample_weight and global_batch_size=1
    per_example_loss = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    sample_weight = None
    global_batch_size = 1
    input_dict = {"per_example_loss": per_example_loss, "sample_weight": sample_weight, "global_batch_size": global_batch_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With sample_weight and global_batch_size
    per_example_loss = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32).numpy()
    sample_weight = tf.constant([0.5, 1.0, 0.5, 1.0], dtype=tf.float32).numpy()
    global_batch_size = 2
    input_dict = {"per_example_loss": per_example_loss, "sample_weight": sample_weight, "global_batch_size": global_batch_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero loss values
    per_example_loss = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32).numpy()
    sample_weight = None
    global_batch_size = 3
    input_dict = {"per_example_loss": per_example_loss, "sample_weight": sample_weight, "global_batch_size": global_batch_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative loss values
    per_example_loss = tf.constant([-1.0, -2.0, -3.0], dtype=tf.float32).numpy()
    sample_weight = None
    global_batch_size = 1
    input_dict = {"per_example_loss": per_example_loss, "sample_weight": sample_weight, "global_batch_size": global_batch_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shape for per_example_loss
    per_example_loss = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32).numpy()
    sample_weight = None
    global_batch_size = 1
    input_dict = {"per_example_loss": per_example_loss, "sample_weight": sample_weight, "global_batch_size": global_batch_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shape for sample_weight
    per_example_loss = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    sample_weight = tf.constant([0.5, 1.0, 0.5], dtype=tf.float32).numpy()
    global_batch_size = 1
    input_dict = {"per_example_loss": per_example_loss, "sample_weight": sample_weight, "global_batch_size": global_batch_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero global_batch_size
    per_example_loss = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    sample_weight = None
    global_batch_size = 0
    input_dict = {"per_example_loss": per_example_loss, "sample_weight": sample_weight, "global_batch_size": global_batch_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: large global_batch_size
    per_example_loss = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    sample_weight = None
    global_batch_size = 100
    input_dict = {"per_example_loss": per_example_loss, "sample_weight": sample_weight, "global_batch_size": global_batch_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: per_example loss as a single value tensor. Fixed to be at least rank 1
    per_example_loss = tf.constant([5.0], dtype=tf.float32).numpy()
    sample_weight = None
    global_batch_size = 1
    input_dict = {"per_example_loss": per_example_loss, "sample_weight": sample_weight, "global_batch_size": global_batch_size}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: All weights are zero
    per_example_loss = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    sample_weight = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32).numpy()
    global_batch_size = 1
    input_dict = {"per_example_loss": per_example_loss, "sample_weight": sample_weight, "global_batch_size": global_batch_size}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.compute_average_loss"] = tf_nn_compute_average_loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.compute_average_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.compute_average_loss'.")

check_valid('tf.nn.compute_average_loss', generated_inputs['tf.nn.compute_average_loss'], lib="tf", suffix=0)
