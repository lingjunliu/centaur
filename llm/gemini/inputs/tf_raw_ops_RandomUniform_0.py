
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_randomuniform_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        'seed': 1,
        'seed2': 1,
        'name': "random_1",
        'shape': np.array([3, 3], dtype=np.int32),
        'dtype': np.float32
    })

    # Input 2
    list_of_inputs.append({
        'seed': 2,
        'seed2': 2,
        'name': "random_2",
        'shape': np.array([10], dtype=np.int64),
        'dtype': np.float64
    })

    # Input 3
    list_of_inputs.append({
        'seed': 3,
        'seed2': 3,
        'name': "random_3",
        'shape': np.array([2, 5, 2], dtype=np.int32),
        'dtype': np.float16
    })

    # Input 4
    list_of_inputs.append({
        'seed': 4,
        'seed2': 4,
        'name': "random_4",
        'shape': np.array([1, 1, 1], dtype=np.int64),
        'dtype': np.float32
    })

    # Input 5
    list_of_inputs.append({
        'seed': 5,
        'seed2': 5,
        'name': "random_5",
        'shape': np.array([8, 8], dtype=np.int32),
        'dtype': np.float64
    })

    # Input 6
    list_of_inputs.append({
        'seed': 6,
        'seed2': 6,
        'name': "random_6",
        'shape': np.array([100, 10], dtype=np.int32),
        'dtype': np.float32
    })

    # Input 7
    list_of_inputs.append({
        'seed': 7,
        'seed2': 7,
        'name': "random_7",
        'shape': np.array([1], dtype=np.int64),
        'dtype': np.float16
    })

    # Input 8
    list_of_inputs.append({
        'seed': 8,
        'seed2': 8,
        'name': "random_8",
        'shape': np.array([4, 4, 4, 4], dtype=np.int32),
        'dtype': np.float32
    })

    # Input 9
    list_of_inputs.append({
        'seed': 9,
        'seed2': 9,
        'name': "random_9",
        'shape': np.array([2, 3], dtype=np.int64),
        'dtype': np.float64
    })

    # Input 10
    list_of_inputs.append({
        'seed': 10,
        'seed2': 10,
        'name': "random_10",
        'shape': np.array([1000], dtype=np.int32),
        'dtype': np.float32
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.RandomUniform"] = tf_raw_ops_randomuniform_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RandomUniform' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomUniform'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.RandomUniform', generated_inputs['tf.raw_ops.RandomUniform'], lib="tf", suffix=0)
