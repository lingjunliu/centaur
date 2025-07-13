
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ParameterizedTruncatedNormal_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([2, 3], dtype=np.int32)
    means = np.array(0.0, dtype=np.float32)
    stdevs = np.array(1.0, dtype=np.float32)
    minvals = np.array(-2.0, dtype=np.float32)
    maxvals = np.array(2.0, dtype=np.float32)
    seed = 0
    seed2 = 0

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([5], dtype=np.int32)
    means = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    stdevs = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    minvals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    maxvals = np.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
    seed = 1
    seed2 = 2

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([1, 2, 3], dtype=np.int32)
    means = np.array(0.5, dtype=np.float64)
    stdevs = np.array(0.2, dtype=np.float64)
    minvals = np.array(0.0, dtype=np.float64)
    maxvals = np.array(1.0, dtype=np.float64)
    seed = 123
    seed2 = 456

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([4, 2], dtype=np.int32)
    means = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    stdevs = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    minvals = np.array([-1.0, -0.5, 0.0, 0.5], dtype=np.float32)
    maxvals = np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    seed = 789
    seed2 = 1011

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    shape = np.array([3, 3], dtype=np.int32)
    means = np.array(1.0, dtype=np.float16)
    stdevs = np.array(0.5, dtype=np.float16)
    minvals = np.array(0.0, dtype=np.float16)
    maxvals = np.array(2.0, dtype=np.float16)
    seed = 0
    seed2 = 0

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([2], dtype=np.int32)
    means = np.array([2.0, 3.0], dtype=np.float64)
    stdevs = np.array([0.1, 0.2], dtype=np.float64)
    minvals = np.array([1.9, 2.8], dtype=np.float64)
    maxvals = np.array([2.1, 3.2], dtype=np.float64)
    seed = 42
    seed2 = 1337

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = np.array([1, 1, 1], dtype=np.int32)
    means = np.array(0.0, dtype=np.float32)
    stdevs = np.array(1.0, dtype=np.float32)
    minvals = np.array(-np.inf, dtype=np.float32)
    maxvals = np.array(np.inf, dtype=np.float32)
    seed = 0
    seed2 = 0

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([2, 2], dtype=np.int64)
    means = np.array([0.5, 1.5], dtype=np.float16)
    stdevs = np.array([0.1, 0.2], dtype=np.float16)
    minvals = np.array([0.4, 1.3], dtype=np.float16)
    maxvals = np.array([0.6, 1.7], dtype=np.float16)
    seed = 10
    seed2 = 20

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = np.array([2], dtype=np.int32)
    means = np.array([0.0, 0.0], dtype=np.float32)
    stdevs = np.array([1.0, 1.0], dtype=np.float32)
    minvals = np.array([-1.0, -1.0], dtype=np.float32)
    maxvals = np.array([1.0, 1.0], dtype=np.float32)
    seed = 1234
    seed2 = 5678

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([1], dtype=np.int64)
    means = np.array([5.0], dtype=np.float64)
    stdevs = np.array([2.0], dtype=np.float64)
    minvals = np.array([1.0], dtype=np.float64)
    maxvals = np.array([6.0], dtype=np.float64)
    seed = 9012
    seed2 = 3456

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ParameterizedTruncatedNormal"] = tf_raw_ops_ParameterizedTruncatedNormal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ParameterizedTruncatedNormal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ParameterizedTruncatedNormal'.")

check_valid('tf.raw_ops.ParameterizedTruncatedNormal', generated_inputs['tf.raw_ops.ParameterizedTruncatedNormal'], lib="tf", suffix=0)
