
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
    means = np.array([0.0], dtype=np.float32)
    stdevs = np.array([1.0], dtype=np.float32)
    minvals = np.array([-1.0], dtype=np.float32)
    maxvals = np.array([1.0], dtype=np.float32)
    seed = 1
    seed2 = 1
    name = None

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([5, 4], dtype=np.int64)
    means = np.array([0.5], dtype=np.float64)
    stdevs = np.array([0.5], dtype=np.float64)
    minvals = np.array([0.0], dtype=np.float64)
    maxvals = np.array([1.0], dtype=np.float64)
    seed = 1
    seed2 = 2
    name = "test2"

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([10], dtype=np.int32)
    means = np.array([0.0] * 1, dtype=np.float32)
    stdevs = np.array([1.0] * 1, dtype=np.float32)
    minvals = np.array([-2.0] * 1, dtype=np.float32)
    maxvals = np.array([2.0] * 1, dtype=np.float32)
    seed = 123
    seed2 = 456
    name = "test3"

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([2, 2], dtype=np.int64)
    means = np.array([0.1], dtype=np.float64)
    stdevs = np.array([0.3], dtype=np.float64)
    minvals = np.array([-0.5], dtype=np.float64)
    maxvals = np.array([0.7], dtype=np.float64)
    seed = 112
    seed2 = 131
    name = "test5"

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    shape = np.array([1], dtype=np.int32)
    means = np.array([1.0], dtype=np.float32)
    stdevs = np.array([2.0], dtype=np.float32)
    minvals = np.array([0.0], dtype=np.float32)
    maxvals = np.array([3.0], dtype=np.float32)
    seed = 42
    seed2 = 24
    name = "test6"

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([4, 4], dtype=np.int64)
    means = np.array([0.0], dtype=np.float64)
    stdevs = np.array([1.0], dtype=np.float64)
    minvals = np.array([-1.0], dtype=np.float64)
    maxvals = np.array([1.0], dtype=np.float64)
    seed = 1000
    seed2 = 2000
    name = "test7"

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = np.array([2, 5], dtype=np.int32)
    means = np.array([0.0], dtype=np.float32)
    stdevs = np.array([1.0], dtype=np.float32)
    minvals = np.array([-3.0], dtype=np.float32)
    maxvals = np.array([3.0], dtype=np.float32)
    seed = 5
    seed2 = 6
    name = "test8"

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([10, 10], dtype=np.int64)
    means = np.array([0.0], dtype=np.float64)
    stdevs = np.array([0.1], dtype=np.float64)
    minvals = np.array([-0.2], dtype=np.float64)
    maxvals = np.array([0.2], dtype=np.float64)
    seed = 7
    seed2 = 8
    name = "test9"

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = np.array([3, 3, 3], dtype=np.int32)
    means = np.array([0.5], dtype=np.float32)
    stdevs = np.array([0.2], dtype=np.float32)
    minvals = np.array([0.0], dtype=np.float32)
    maxvals = np.array([1.0], dtype=np.float32)
    seed = 9
    seed2 = 10
    name = "test10"

    input_dict = {
        "shape": shape,
        "means": means,
        "stdevs": stdevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "seed": seed,
        "seed2": seed2,
        "name": name
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
