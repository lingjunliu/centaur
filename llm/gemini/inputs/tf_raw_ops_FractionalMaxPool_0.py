
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_FractionalMaxPool_inputs():
    list_of_inputs = []

    # Input 1
    value = np.arange(36, dtype=np.float32).reshape((1, 6, 6, 1))
    pooling_ratio = [1.0, 1.5, 1.5, 1.0]
    input_dict = {
        'pseudo_random': False,
        'overlapping': False,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': "pool1",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.random.uniform(-10, 10, (1, 10, 10, 3)).astype(np.float64)
    pooling_ratio = [1.0, 2.0, 2.0, 1.0]
    input_dict = {
        'pseudo_random': True,
        'overlapping': True,
        'deterministic': True,
        'seed': 42,
        'seed2': 24,
        'name': "pool2",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.random.randint(0, 100, (2, 8, 8, 2), dtype=np.int32)
    pooling_ratio = [1.0, 1.2, 1.2, 1.0]
    input_dict = {
        'pseudo_random': True,
        'overlapping': False,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': "pool3",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.random.randint(-50, 50, (1, 12, 12, 1), dtype=np.int64)
    pooling_ratio = [1.0, 1.44, 1.73, 1.0]
    input_dict = {
        'pseudo_random': False,
        'overlapping': True,
        'deterministic': True,
        'seed': 1,
        'seed2': 2,
        'name': "pool4",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.random.normal(0, 1, (2, 5, 5, 1)).astype(np.float32)
    pooling_ratio = [1.0, 1.1, 1.1, 1.0]
    input_dict = {
        'pseudo_random': True,
        'overlapping': True,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': "pool5",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    value = np.ones((1, 15, 15, 4), dtype=np.float32) * 5.5
    pooling_ratio = [1.0, 1.8, 1.3, 1.0]
    input_dict = {
        'pseudo_random': False,
        'overlapping': False,
        'deterministic': True,
        'seed': 7,
        'seed2': 14,
        'name': "pool6",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    value = np.random.randint(-10, 10, (3, 6, 6, 3), dtype=np.int32)
    pooling_ratio = [1.0, 1.5, 1.2, 1.0]
    input_dict = {
        'pseudo_random': True,
        'overlapping': False,
        'deterministic': True,
        'seed': 99,
        'seed2': 99,
        'name': "pool7",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    value = np.random.uniform(-1, 1, (1, 4, 4, 1)).astype(np.float64)
    pooling_ratio = [1.0, 1.0, 1.0, 1.0]
    input_dict = {
        'pseudo_random': False,
        'overlapping': True,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': "pool8",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    value = np.random.randint(0, 10, (4, 10, 10, 2), dtype=np.int64)
    pooling_ratio = [1.0, 2.5, 2.5, 1.0]
    input_dict = {
        'pseudo_random': False,
        'overlapping': False,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': "pool9",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.random.normal(5, 2, (2, 20, 20, 3)).astype(np.float32)
    pooling_ratio = [1.0, 3.0, 1.5, 1.0]
    input_dict = {
        'pseudo_random': True,
        'overlapping': True,
        'deterministic': True,
        'seed': 12345,
        'seed2': 54321,
        'name': "pool10",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FractionalMaxPool"] = tf_raw_ops_FractionalMaxPool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FractionalMaxPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FractionalMaxPool'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.FractionalMaxPool', generated_inputs['tf.raw_ops.FractionalMaxPool'], lib="tf", suffix=0)
