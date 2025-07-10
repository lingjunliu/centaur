
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_random_normal_initializer_inputs():
    list_of_inputs = []

    # Input 1
    mean = np.float32(0.0)
    stddev = np.float32(0.05)
    seed = None

    input_dict = {
        "mean": mean,
        "stddev": stddev,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    mean = np.float32(1.0)
    stddev = np.float32(2.0)
    seed = 123

    input_dict = {
        "mean": mean,
        "stddev": stddev,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    mean = np.float32(-1.0)
    stddev = np.float32(0.5)
    seed = 42

    input_dict = {
        "mean": mean,
        "stddev": stddev,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    mean = np.float32(0.5)
    stddev = np.float32(1.5)
    seed = 0

    input_dict = {
        "mean": mean,
        "stddev": stddev,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    mean = np.float32(-0.5)
    stddev = np.float32(0.25)
    seed = -1

    input_dict = {
        "mean": mean,
        "stddev": stddev,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    mean = np.float32(10.0)
    stddev = np.float32(5.0)
    seed = 1000

    input_dict = {
        "mean": mean,
        "stddev": stddev,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    mean = np.float32(-5.0)
    stddev = np.float32(2.5)
    seed = -500

    input_dict = {
        "mean": mean,
        "stddev": stddev,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    mean = np.float32(0.1)
    stddev = np.float32(0.01)
    seed = 1

    input_dict = {
        "mean": mean,
        "stddev": stddev,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    mean = np.float32(-0.1)
    stddev = np.float32(0.005)
    seed = -10

    input_dict = {
        "mean": mean,
        "stddev": stddev,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    mean = np.float32(100.0)
    stddev = np.float32(10.0)
    seed = 9999

    input_dict = {
        "mean": mean,
        "stddev": stddev,
        "seed": seed
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
