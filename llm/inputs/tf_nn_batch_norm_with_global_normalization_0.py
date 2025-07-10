
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_batch_norm_with_global_normalization_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 28, 28, 3).astype(np.float32)
    mean_tensor = np.random.rand(3).astype(np.float32)
    variance_tensor = np.random.rand(3).astype(np.float32)
    beta_tensor = np.random.rand(3).astype(np.float32)
    gamma_tensor = np.random.rand(3).astype(np.float32)
    variance_epsilon_val = 0.001
    scale_after_normalization_val = True
    name_val = "batch_norm_1"

    input_dict = {
        'input': input_tensor,
        'mean': mean_tensor,
        'variance': variance_tensor,
        'beta': beta_tensor,
        'gamma': gamma_tensor,
        'variance_epsilon': variance_epsilon_val,
        'scale_after_normalization': scale_after_normalization_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(4, 16, 16, 1).astype(np.float32)
    mean_tensor = np.random.rand(1).astype(np.float32)
    variance_tensor = np.random.rand(1).astype(np.float32)
    beta_tensor = np.random.rand(1).astype(np.float32)
    gamma_tensor = np.random.rand(1).astype(np.float32)
    variance_epsilon_val = 0.00001
    scale_after_normalization_val = False
    name_val = "batch_norm_2"

    input_dict = {
        'input': input_tensor,
        'mean': mean_tensor,
        'variance': variance_tensor,
        'beta': beta_tensor,
        'gamma': gamma_tensor,
        'variance_epsilon': variance_epsilon_val,
        'scale_after_normalization': scale_after_normalization_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(2, 64, 64, 32).astype(np.float32)
    mean_tensor = np.random.rand(32).astype(np.float32)
    variance_tensor = np.random.rand(32).astype(np.float32)
    beta_tensor = np.random.rand(32).astype(np.float32)
    gamma_tensor = np.random.rand(32).astype(np.float32)
    variance_epsilon_val = 0.1
    scale_after_normalization_val = True
    name_val = None

    input_dict = {
        'input': input_tensor,
        'mean': mean_tensor,
        'variance': variance_tensor,
        'beta': beta_tensor,
        'gamma': gamma_tensor,
        'variance_epsilon': variance_epsilon_val,
        'scale_after_normalization': scale_after_normalization_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    input_tensor = np.random.rand(8, 32, 32, 16).astype(np.float32)
    mean_tensor = np.random.rand(16).astype(np.float32)
    variance_tensor = np.random.rand(16).astype(np.float32)
    beta_tensor = np.random.rand(16).astype(np.float32)
    gamma_tensor = np.random.rand(16).astype(np.float32)
    variance_epsilon_val = 0.01
    scale_after_normalization_val = False
    name_val = "batch_norm_4"

    input_dict = {
        'input': input_tensor,
        'mean': mean_tensor,
        'variance': variance_tensor,
        'beta': beta_tensor,
        'gamma': gamma_tensor,
        'variance_epsilon': variance_epsilon_val,
        'scale_after_normalization': scale_after_normalization_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 128, 128, 64).astype(np.float32)
    mean_tensor = np.random.rand(64).astype(np.float32)
    variance_tensor = np.random.rand(64).astype(np.float32)
    beta_tensor = np.random.rand(64).astype(np.float32)
    gamma_tensor = np.random.rand(64).astype(np.float32)
    variance_epsilon_val = 0.0001
    scale_after_normalization_val = True
    name_val = "batch_norm_5"

    input_dict = {
        'input': input_tensor,
        'mean': mean_tensor,
        'variance': variance_tensor,
        'beta': beta_tensor,
        'gamma': gamma_tensor,
        'variance_epsilon': variance_epsilon_val,
        'scale_after_normalization': scale_after_normalization_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(16, 8, 8, 4).astype(np.float32)
    mean_tensor = np.random.rand(4).astype(np.float32)
    variance_tensor = np.random.rand(4).astype(np.float32)
    beta_tensor = np.random.rand(4).astype(np.float32)
    gamma_tensor = np.random.rand(4).astype(np.float32)
    variance_epsilon_val = 1e-08
    scale_after_normalization_val = False
    name_val = "batch_norm_6"

    input_dict = {
        'input': input_tensor,
        'mean': mean_tensor,
        'variance': variance_tensor,
        'beta': beta_tensor,
        'gamma': gamma_tensor,
        'variance_epsilon': variance_epsilon_val,
        'scale_after_normalization': scale_after_normalization_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(3, 256, 256, 128).astype(np.float32)
    mean_tensor = np.random.rand(128).astype(np.float32)
    variance_tensor = np.random.rand(128).astype(np.float32)
    beta_tensor = np.random.rand(128).astype(np.float32)
    gamma_tensor = np.random.rand(128).astype(np.float32)
    variance_epsilon_val = 1e-05
    scale_after_normalization_val = True
    name_val = None

    input_dict = {
        'input': input_tensor,
        'mean': mean_tensor,
        'variance': variance_tensor,
        'beta': beta_tensor,
        'gamma': gamma_tensor,
        'variance_epsilon': variance_epsilon_val,
        'scale_after_normalization': scale_after_normalization_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: negative values
    input_tensor = np.random.rand(1, 28, 28, 3).astype(np.float32) - 0.5
    mean_tensor = np.random.rand(3).astype(np.float32) - 0.5
    variance_tensor = np.random.rand(3).astype(np.float32)
    beta_tensor = np.random.rand(3).astype(np.float32) - 0.5
    gamma_tensor = np.random.rand(3).astype(np.float32)
    variance_epsilon_val = 0.001
    scale_after_normalization_val = True
    name_val = "batch_norm_8"

    input_dict = {
        'input': input_tensor,
        'mean': mean_tensor,
        'variance': variance_tensor,
        'beta': beta_tensor,
        'gamma': gamma_tensor,
        'variance_epsilon': variance_epsilon_val,
        'scale_after_normalization': scale_after_normalization_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(1, 5, 5, 2).astype(np.float32)
    mean_tensor = np.random.rand(2).astype(np.float32)
    variance_tensor = np.random.rand(2).astype(np.float32)
    beta_tensor = np.random.rand(2).astype(np.float32)
    gamma_tensor = np.random.rand(2).astype(np.float32)
    variance_epsilon_val = 0.0001
    scale_after_normalization_val = False
    name_val = "batch_norm_9"

    input_dict = {
        'input': input_tensor,
        'mean': mean_tensor,
        'variance': variance_tensor,
        'beta': beta_tensor,
        'gamma': gamma_tensor,
        'variance_epsilon': variance_epsilon_val,
        'scale_after_normalization': scale_after_normalization_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(2, 10, 10, 8).astype(np.float32)
    mean_tensor = np.random.rand(8).astype(np.float32)
    variance_tensor = np.random.rand(8).astype(np.float32)
    beta_tensor = np.random.rand(8).astype(np.float32)
    gamma_tensor = np.random.rand(8).astype(np.float32)
    variance_epsilon_val = 0.001
    scale_after_normalization_val = True
    name_val = "batch_norm_10"

    input_dict = {
        'input': input_tensor,
        'mean': mean_tensor,
        'variance': variance_tensor,
        'beta': beta_tensor,
        'gamma': gamma_tensor,
        'variance_epsilon': variance_epsilon_val,
        'scale_after_normalization': scale_after_normalization_val,
        'name': name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.batch_norm_with_global_normalization"] = tf_nn_batch_norm_with_global_normalization_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.batch_norm_with_global_normalization' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.batch_norm_with_global_normalization'.")

check_valid('tf.nn.batch_norm_with_global_normalization', generated_inputs['tf.nn.batch_norm_with_global_normalization'], lib="tf", suffix=0)
