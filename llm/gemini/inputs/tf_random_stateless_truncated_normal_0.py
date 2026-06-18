
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_truncated_normal_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'shape': np.array([2, 3], dtype=np.int32),
        'seed': np.array([1, 2], dtype=np.int32),
        'mean': 0.0,
        'stddev': 1.0,
        'dtype': np.float32,
        'name': "truncated_normal_1",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'shape': np.array([5], dtype=np.int64),
        'seed': np.array([42, 24], dtype=np.int64),
        'mean': -1.5,
        'stddev': 0.5,
        'dtype': np.float64,
        'name': "truncated_normal_2",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'shape': np.array([2, 2, 2], dtype=np.int32),
        'seed': np.array([7, 8], dtype=np.int32),
        'mean': 10.0,
        'stddev': 2.5,
        'dtype': np.float32,
        'name': "truncated_normal_3",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'shape': np.array([10], dtype=np.int32),
        'seed': np.array([0, 0], dtype=np.int32),
        'mean': 0.0,
        'stddev': 0.001,
        'dtype': np.float16,
        'name': "truncated_normal_4",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'shape': np.array([3, 1], dtype=np.int32),
        'seed': np.array([123, 456], dtype=np.int32),
        'mean': -100.0,
        'stddev': 50.0,
        'dtype': np.float64,
        'name': "truncated_normal_5",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'shape': np.array([1], dtype=np.int32),
        'seed': np.array([9, 9], dtype=np.int32),
        'mean': 0.5,
        'stddev': 0.1,
        'dtype': np.float32,
        'name': "truncated_normal_6",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'shape': np.array([1, 5, 1], dtype=np.int64),
        'seed': np.array([100, 200], dtype=np.int64),
        'mean': 3.14,
        'stddev': 1.59,
        'dtype': np.float32,
        'name': "truncated_normal_7",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'shape': np.array([4, 4], dtype=np.int32),
        'seed': np.array([11, 22], dtype=np.int32),
        'mean': -0.01,
        'stddev': 0.02,
        'dtype': np.float64,
        'name': "truncated_normal_8",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'shape': np.array([2, 3, 4], dtype=np.int32),
        'seed': np.array([1234, 5678], dtype=np.int32),
        'mean': 12.3,
        'stddev': 4.5,
        'dtype': np.float16,
        'name': "truncated_normal_9",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'shape': np.array([8], dtype=np.int64),
        'seed': np.array([999, 888], dtype=np.int64),
        'mean': -5.0,
        'stddev': 0.01,
        'dtype': np.float64,
        'name': "truncated_normal_10",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_truncated_normal"] = tf_random_stateless_truncated_normal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_truncated_normal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_truncated_normal'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_truncated_normal', generated_inputs['tf.random.stateless_truncated_normal'], lib="tf", suffix=0)
