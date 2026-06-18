
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_TruncatedNormal_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        'shape': np.array([2, 3], dtype=np.int32),
        'dtype': np.float32,
        'seed': 1,
        'seed2': 2,
        'name': "truncated_normal_1"
    })

    # Input 2
    list_of_inputs.append({
        'shape': np.array([5], dtype=np.int64),
        'dtype': np.float64,
        'seed': 3,
        'seed2': 4,
        'name': "truncated_normal_2"
    })

    # Input 3
    list_of_inputs.append({
        'shape': np.array([3, 3, 3], dtype=np.int32),
        'dtype': np.float16,
        'seed': 42,
        'seed2': 123,
        'name': "truncated_normal_3"
    })

    # Input 4
    list_of_inputs.append({
        'shape': np.array([1], dtype=np.int32),
        'dtype': np.float32,
        'seed': 10,
        'seed2': 20,
        'name': "truncated_normal_4"
    })

    # Input 5
    list_of_inputs.append({
        'shape': np.array([10, 10], dtype=np.int64),
        'dtype': np.float32,
        'seed': 100,
        'seed2': 200,
        'name': "truncated_normal_5"
    })

    # Input 6
    list_of_inputs.append({
        'shape': np.array([1, 2, 3, 4], dtype=np.int32),
        'dtype': np.float64,
        'seed': 7,
        'seed2': 14,
        'name': "truncated_normal_6"
    })

    # Input 7
    list_of_inputs.append({
        'shape': np.array([], dtype=np.int32),
        'dtype': np.float32,
        'seed': 5,
        'seed2': 1,
        'name': "truncated_normal_7"
    })

    # Input 8
    list_of_inputs.append({
        'shape': np.array([100], dtype=np.int32),
        'dtype': np.float16,
        'seed': 999,
        'seed2': 888,
        'name': "truncated_normal_8"
    })

    # Input 9
    list_of_inputs.append({
        'shape': np.array([2, 2, 2, 2, 2], dtype=np.int64),
        'dtype': np.float32,
        'seed': 12345,
        'seed2': 54321,
        'name': "truncated_normal_9"
    })

    # Input 10
    list_of_inputs.append({
        'shape': np.array([4, 1], dtype=np.int32),
        'dtype': np.float64,
        'seed': 12,
        'seed2': 34,
        'name': "truncated_normal_10"
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.TruncatedNormal"] = tf_raw_ops_TruncatedNormal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.TruncatedNormal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TruncatedNormal'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.TruncatedNormal', generated_inputs['tf.raw_ops.TruncatedNormal'], lib="tf", suffix=0)
