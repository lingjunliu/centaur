
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_gamma_inputs():
    list_of_inputs = []

    # Input 1
    input_dict_1 = {
        "shape": np.array([5], dtype=np.int32),
        "seed": np.array([42, 43], dtype=np.int32),
        "alpha": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "beta": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2
    input_dict_2 = {
        "shape": np.array([10, 2], dtype=np.int32),
        "seed": np.array([12, 34], dtype=np.int32),
        "alpha": np.array([0.5, 1.5], dtype=np.float32),
        "beta": np.array([1.0, 2.0], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3
    input_dict_3 = {
        "shape": np.array([3, 4], dtype=np.int32),
        "seed": np.array([7, 8], dtype=np.int64),
        "alpha": np.array([[1.0], [2.0], [3.0]], dtype=np.float64),
        "beta": np.array([[2.0, 3.0, 4.0, 5.0]], dtype=np.float64),
        "dtype": np.float64,
        "name": "gamma_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4
    input_dict_4 = {
        "shape": np.array([5, 3, 4], dtype=np.int32),
        "seed": np.array([1, 2], dtype=np.int32),
        "alpha": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        "beta": np.array([[2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5
    input_dict_5 = {
        "shape": np.array([1], dtype=np.int32),
        "seed": np.array([100, 200], dtype=np.int32),
        "alpha": np.array([2.5], dtype=np.float32),
        "beta": np.array([1.0], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6
    input_dict_6 = {
        "shape": np.array([2, 3], dtype=np.int32),
        "seed": np.array([999, 888], dtype=np.int32),
        "alpha": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "beta": np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_large_alpha"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7
    input_dict_7 = {
        "shape": np.array([4, 2, 3], dtype=np.int32),
        "seed": np.array([11, 22], dtype=np.int32),
        "alpha": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16),
        "beta": np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float16),
        "dtype": np.float16,
        "name": "gamma_float16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8
    input_dict_8 = {
        "shape": np.array([2, 2], dtype=np.int32),
        "seed": np.array([3, 4], dtype=np.int64),
        "alpha": np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
        "beta": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_4d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9
    input_dict_9 = {
        "shape": np.array([1, 1], dtype=np.int32),
        "seed": np.array([123, 456], dtype=np.int32),
        "alpha": np.array([[1.0]], dtype=np.float32),
        "beta": np.array([[1.0]], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_scalar"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10
    input_dict_10 = {
        "shape": np.array([2, 1, 3], dtype=np.int32),
        "seed": np.array([123456789, 987654321], dtype=np.int64),
        "alpha": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "beta": np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_large_seed"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.random.stateless_gamma"] = tf_random_stateless_gamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_gamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_gamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_gamma', generated_inputs['tf.random.stateless_gamma'], lib="tf", suffix=0)
