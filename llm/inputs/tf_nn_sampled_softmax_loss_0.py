
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_nn_sampled_softmax_loss_inputs():
    list_of_inputs = []

    # The `sampled_values` key is completely removed from all inputs.
    # The API will use its default sampler, which is the desired behavior and
    # avoids the testing framework's errors with None/tuple values.

    # Input 1: Basic case with float32 and num_true=1
    input_dict_1 = {
        'weights': np.random.randn(100, 10).astype(np.float32),
        'biases': np.random.randn(100).astype(np.float32),
        'labels': np.random.randint(0, 100, size=(4, 1), dtype=np.int64),
        'inputs': np.random.randn(4, 10).astype(np.float32),
        'num_sampled': 5,
        'num_classes': 100,
        'num_true': 1,
        'remove_accidental_hits': True,
        'seed': 123,
        'name': 'basic_case_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multi-label (num_true > 1), float64, and remove_accidental_hits=False
    input_dict_2 = {
        'weights': np.random.randn(1000, 16).astype(np.float64),
        'biases': np.random.randn(1000).astype(np.float64),
        'labels': np.random.randint(0, 1000, size=(8, 3), dtype=np.int64),
        'inputs': np.random.randn(8, 16).astype(np.float64),
        'num_sampled': 20,
        'num_classes': 1000,
        'num_true': 3,
        'remove_accidental_hits': False,
        'seed': 456,
        'name': 'multi_label_float64_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Small num_classes and large sampling ratio
    input_dict_3 = {
        'weights': np.random.randn(12, 4).astype(np.float32),
        'biases': np.random.randn(12).astype(np.float32),
        'labels': np.random.randint(0, 12, size=(5, 1), dtype=np.int64),
        'inputs': np.random.randn(5, 4).astype(np.float32),
        'num_sampled': 10,
        'num_classes': 12,
        'num_true': 1,
        'remove_accidental_hits': False,
        'seed': 101,
        'name': 'large_sample_ratio_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Large dimensions
    input_dict_4 = {
        'weights': np.random.randn(500, 128).astype(np.float32),
        'biases': np.random.randn(500).astype(np.float32),
        'labels': np.random.randint(0, 500, size=(16, 1), dtype=np.int64),
        'inputs': np.random.randn(16, 128).astype(np.float32),
        'num_sampled': 64,
        'num_classes': 500,
        'num_true': 1,
        'remove_accidental_hits': True,
        'seed': 111,
        'name': 'large_dims_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Minimal valid case
    input_dict_5 = {
        'weights': np.array([[0.1], [-0.2]], dtype=np.float32),
        'biases': np.array([0.5, -0.5], dtype=np.float32),
        'labels': np.array([[1]], dtype=np.int64),
        'inputs': np.array([[0.3]], dtype=np.float32),
        'num_sampled': 1,
        'num_classes': 2,
        'num_true': 1,
        'remove_accidental_hits': True,
        'seed': 303,
        'name': 'minimal_case_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Zero seed
    input_dict_6 = {
        'weights': np.random.randn(20, 5).astype(np.float32),
        'biases': np.random.randn(20).astype(np.float32),
        'labels': np.random.randint(0, 20, size=(3, 1), dtype=np.int64),
        'inputs': np.random.randn(3, 5).astype(np.float32),
        'num_sampled': 4,
        'num_classes': 20,
        'num_true': 1,
        'remove_accidental_hits': True,
        'seed': 0,
        'name': 'zero_seed_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Large num_true
    input_dict_7 = {
        'weights': np.random.randn(1000, 64).astype(np.float32),
        'biases': np.random.randn(1000).astype(np.float32),
        'labels': np.random.randint(0, 1000, size=(4, 10), dtype=np.int64),
        'inputs': np.random.randn(4, 64).astype(np.float32),
        'num_sampled': 50,
        'num_classes': 1000,
        'num_true': 10,
        'remove_accidental_hits': True,
        'seed': 999,
        'name': 'large_num_true_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs["tf.nn.sampled_softmax_loss"] = tf_nn_sampled_softmax_loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.sampled_softmax_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.sampled_softmax_loss'.")

check_valid('tf.nn.sampled_softmax_loss', generated_inputs['tf.nn.sampled_softmax_loss'], lib="tf", suffix=0)
