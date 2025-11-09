
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

np.random.seed(42)

def tf_nn_log_poisson_loss_inputs():
    list_of_inputs = []

    targets = np.array(3.0, dtype=np.float32)
    log_input = np.array(np.log(2.5), dtype=np.float32)
    compute_full_loss = False
    name = "scalar_f32_no_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    log_input = np.array([np.log(0.5), np.log(1.5), np.log(3.0)], dtype=np.float64)
    compute_full_loss = False
    name = "vector_f64_no_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    log_input = np.array([[0.0, -0.7], [1.2, 2.0]], dtype=np.float32)
    compute_full_loss = True
    name = "matrix2x2_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = (np.random.rand(2, 3, 4) * 3.0).astype(np.float64)
    log_input = np.random.randn(2, 3, 4).astype(np.float64)
    compute_full_loss = False
    name = "tensor3d_f64_no_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = (np.random.rand(1, 2, 2, 3).astype(np.float32) + 0.1)
    log_input = np.random.randn(1, 2, 2, 3).astype(np.float32)
    compute_full_loss = True
    name = "tensor4d_f32_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([0.0, 0.0, 1.0, 2.0, 5.0], dtype=np.float32)
    log_input = np.array([-2.0, 0.0, 0.5, 1.0, 2.3], dtype=np.float32)
    compute_full_loss = False
    name = "vector_f32_zeros_no_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([[0.1], [2.0]], dtype=np.float64)
    log_input = np.array([[np.log(0.3)], [np.log(5.0)]], dtype=np.float64)
    compute_full_loss = True
    name = "column_f64_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([], dtype=np.float32)
    log_input = np.array([], dtype=np.float32)
    compute_full_loss = False
    name = "empty_f32_no_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    rates = np.array([[0.8, 1.2, 2.5], [0.3, 4.0, 0.1], [1.5, 2.2, 3.3]], dtype=np.float64)
    targets = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=np.float64)
    log_input = np.log(rates)
    compute_full_loss = False
    name = "matrix3x3_f64_no_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([50.0, 1000.0, 100000.0], dtype=np.float64)
    log_input = np.log(np.array([60.0, 900.0, 100000.0], dtype=np.float64))
    compute_full_loss = True
    name = "large_counts_f64_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([-1.0, 0.0], dtype=np.float32)
    log_input = np.array([np.log(1.0), np.log(2.0)], dtype=np.float32)
    compute_full_loss = False
    name = "neg_target_f32_no_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array(0.7, dtype=np.float64)
    log_input = np.array(-0.3, dtype=np.float64)
    compute_full_loss = True
    name = "scalar_f64_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.log_poisson_loss"] = tf_nn_log_poisson_loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.log_poisson_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.log_poisson_loss'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.log_poisson_loss', generated_inputs['tf.nn.log_poisson_loss'], lib="tf", suffix=0)
