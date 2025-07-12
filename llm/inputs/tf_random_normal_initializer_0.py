
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_random_normal_initializer_inputs():
    list_of_inputs = []

    # Input 1
    mean = 0.0
    stddev = 0.05
    seed = None

    initializer = tf.random_normal_initializer(mean=mean, stddev=stddev, seed=seed)
    input_dict = {
        "shape": [2,3],
        "dtype": tf.float32,
        "initializer": initializer
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    mean = 1.0
    stddev = 0.1
    seed = 123

    initializer = tf.random_normal_initializer(mean=mean, stddev=stddev, seed=seed)

    input_dict = {
        "shape": [5, 5],
        "dtype": tf.float64,
        "initializer": initializer
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    mean = -1.0
    stddev = 0.2
    seed = 456
    initializer = tf.random_normal_initializer(mean=mean, stddev=stddev, seed=seed)

    input_dict = {
        "shape": [10],
        "dtype": tf.float32,
        "initializer": initializer
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    mean = 0.5
    stddev = 0.01
    seed = 789
    initializer = tf.random_normal_initializer(mean=mean, stddev=stddev, seed=seed)

    input_dict = {
        "shape": [1, 1, 1],
        "dtype": tf.float64,
        "initializer": initializer
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    mean = -0.5
    stddev = 0.02
    seed = 101
    initializer = tf.random_normal_initializer(mean=mean, stddev=stddev, seed=seed)

    input_dict = {
        "shape": [4, 4, 4],
        "dtype": tf.float32,
        "initializer": initializer
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    mean = 2.0
    stddev = 0.5
    seed = 202
    initializer = tf.random_normal_initializer(mean=mean, stddev=stddev, seed=seed)

    input_dict = {
        "shape": [2, 2, 2, 2],
        "dtype": tf.float64,
        "initializer": initializer
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    mean = -2.0
    stddev = 0.7
    seed = 303
    initializer = tf.random_normal_initializer(mean=mean, stddev=stddev, seed=seed)

    input_dict = {
        "shape": [16],
        "dtype": tf.float32,
        "initializer": initializer
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    mean = 0.123
    stddev = 0.045
    seed = 404
    initializer = tf.random_normal_initializer(mean=mean, stddev=stddev, seed=seed)
    input_dict = {
        "shape": [8, 8],
        "dtype": tf.float64,
        "initializer": initializer
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    mean = -0.345
    stddev = 0.067
    seed = 505
    initializer = tf.random_normal_initializer(mean=mean, stddev=stddev, seed=seed)

    input_dict = {
        "shape": [3, 3, 3],
        "dtype": tf.float32,
        "initializer": initializer
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    mean = 1.5
    stddev = 0.3
    seed = 606
    initializer = tf.random_normal_initializer(mean=mean, stddev=stddev, seed=seed)

    input_dict = {
        "shape": [6, 6, 6, 6],
        "dtype": tf.float64,
        "initializer": initializer
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random_normal_initializer"] = tf_random_normal_initializer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random_normal_initializer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random_normal_initializer'.")

check_valid('tf.random_normal_initializer', generated_inputs['tf.random_normal_initializer'], lib="tf", suffix=0)
