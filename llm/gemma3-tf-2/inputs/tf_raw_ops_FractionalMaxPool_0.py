
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_FractionalMaxPool_inputs():
    list_of_inputs = []

    input_1 = {
        'value': np.random.rand(1, 8, 8, 3).astype(np.float32),
        'pooling_ratio': [1.0, 1.414, 1.732, 1.0],
        'pseudo_random': True,
        'overlapping': False,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': 'fractional_max_pool_1'
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    input_2 = {
        'value': np.random.randint(-10, 10, size=(1, 16, 16, 2)).astype(np.int32),
        'pooling_ratio': [1.0, 1.5, 1.5, 1.0],
        'pseudo_random': False,
        'overlapping': True,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': 'fractional_max_pool_2'
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    input_3 = {
        'value': np.random.rand(2, 32, 32, 4).astype(np.float64),
        'pooling_ratio': [1.0, 2.0, 2.0, 1.0],
        'pseudo_random': True,
        'overlapping': False,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': 'fractional_max_pool_3'
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    input_4 = {
        'value': np.random.randint(0, 255, size=(1, 64, 64, 1)).astype(np.int64),
        'pooling_ratio': [1.0, 1.2, 1.2, 1.0],
        'pseudo_random': False,
        'overlapping': True,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': 'fractional_max_pool_4'
    }
    list_of_inputs.append(copy.deepcopy(input_4))

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
