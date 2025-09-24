
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_log_poisson_loss_inputs():
    list_of_inputs = []

    # Input 1: Basic case, compute_full_loss=False
    targets = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    log_input = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    compute_full_loss = False
    name = "loss1"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: compute_full_loss=True
    targets = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    log_input = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    compute_full_loss = True
    name = "loss2"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative log_input values
    targets = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    log_input = np.array([-0.5, -1.0, -1.5], dtype=np.float32)
    compute_full_loss = False
    name = "loss3"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero targets
    targets = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    log_input = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    compute_full_loss = False
    name = "loss4"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Higher dimension
    targets = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    log_input = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    compute_full_loss = False
    name = "loss5"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: Higher dimension, compute_full_loss=True
    targets = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    log_input = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    compute_full_loss = True
    name = "loss6"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different values for targets
    targets = np.array([5.0, 10.0, 15.0], dtype=np.float32)
    log_input = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    compute_full_loss = False
    name = "loss7"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different values for log_input
    targets = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    log_input = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    compute_full_loss = False
    name = "loss8"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float64
    targets = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    log_input = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    compute_full_loss = False
    name = "loss9"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float64 and compute_full_loss
    targets = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    log_input = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    compute_full_loss = True
    name = "loss10"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.log_poisson_loss"] = tf_nn_log_poisson_loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.log_poisson_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.log_poisson_loss'.")

check_valid('tf.nn.log_poisson_loss', generated_inputs['tf.nn.log_poisson_loss'], lib="tf", suffix=0)
