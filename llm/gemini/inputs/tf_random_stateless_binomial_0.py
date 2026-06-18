
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_binomial_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'shape': np.array([2], dtype=np.int32),
        'seed': np.array([123, 456], dtype=np.int32),
        'counts': np.array([10., 20.], dtype=np.float32),
        'probs': np.array([0.8, 0.5], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'binom_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'shape': np.array([5], dtype=np.int32),
        'seed': np.array([789, 1011], dtype=np.int32),
        'counts': np.array([10., 10., 10., 10., 10.], dtype=np.float32),
        'probs': np.array([0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        'output_dtype': np.int64,
        'name': 'binom_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'shape': np.array([3, 2], dtype=np.int32),
        'seed': np.array([1, 2], dtype=np.int64),
        'counts': np.array([[5., 15.], [5., 15.], [5., 15.]], dtype=np.float32),
        'probs': np.array([[0.2, 0.7], [0.2, 0.7], [0.2, 0.7]], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'binom_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'shape': np.array([4, 1], dtype=np.int32),
        'seed': np.array([42, 42], dtype=np.int32),
        'counts': np.array([[100.], [100.], [100.], [100.]], dtype=np.float32),
        'probs': np.array([[0.5], [0.5], [0.5], [0.5]], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'binom_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'shape': np.array([3], dtype=np.int64),
        'seed': np.array([-1, -2], dtype=np.int32),
        'counts': np.array([50., 50., 50.], dtype=np.float64),
        'probs': np.array([0.1, 0.2, 0.3], dtype=np.float64),
        'output_dtype': np.int64,
        'name': 'binom_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'shape': np.array([2, 2], dtype=np.int32),
        'seed': np.array([0, 0], dtype=np.int32),
        'counts': np.array([[10., 20.], [30., 40.]], dtype=np.float32),
        'probs': np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'binom_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'shape': np.array([1], dtype=np.int32),
        'seed': np.array([1000, 2000], dtype=np.int64),
        'counts': np.array([1.], dtype=np.float32),
        'probs': np.array([0.99], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'binom_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'shape': np.array([4], dtype=np.int32),
        'seed': np.array([11, 22], dtype=np.int32),
        'counts': np.array([5., 10., 15., 20.], dtype=np.float32),
        'probs': np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        'output_dtype': np.int64,
        'name': 'binom_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'shape': np.array([2, 3], dtype=np.int32),
        'seed': np.array([99, 99], dtype=np.int32),
        'counts': np.array([[10., 20., 30.], [40., 50., 60.]], dtype=np.float32),
        'probs': np.array([[0.4, 0.4, 0.4], [0.4, 0.4, 0.4]], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'binom_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'shape': np.array([1, 1], dtype=np.int32),
        'seed': np.array([8888, 9999], dtype=np.int64),
        'counts': np.array([[5.]], dtype=np.float32),
        'probs': np.array([[0.0]], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'binom_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_binomial"] = tf_random_stateless_binomial_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_binomial' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_binomial'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_binomial', generated_inputs['tf.random.stateless_binomial'], lib="tf", suffix=0)
