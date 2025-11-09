
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_parameterized_truncated_normal_inputs():
    list_of_inputs = []

    # Input 1
    shape = [10]
    seed = np.array([123, 456], dtype=np.int32)
    means = np.array(0.0, dtype=np.float32)
    stddevs = np.array(1.0, dtype=np.float32)
    minvals = np.array(-2.0, dtype=np.float32)
    maxvals = np.array(2.0, dtype=np.float32)
    name = "case1"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = [5, 3]
    seed = np.array([2021, 7], dtype=np.int64)
    means = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    stddevs = np.array([0.5, 2.0, 1.5], dtype=np.float64)
    minvals = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    maxvals = np.array([1.0, 4.0, 2.0], dtype=np.float64)
    name = "case2"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = [4, 2, 3]
    seed = np.array([11, 22], dtype=np.int32)
    means = np.array(0.5, dtype=np.float16)
    stddevs = np.array(0.3, dtype=np.float16)
    minvals = np.array([[-1.0, -0.5, -2.0], [-1.5, -0.1, -1.0]], dtype=np.float16)
    maxvals = np.array([[1.0, 0.7, 2.0], [1.2, 0.8, 1.5]], dtype=np.float16)
    name = "case3"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = [2, 3, 1, 4]
    seed = np.array([314, 159], dtype=np.int64)
    means = np.array(np.reshape(np.linspace(-1.0, 1.0, 12, dtype=np.float32), (3, 1, 4)), dtype=np.float32)
    stddevs = np.array(np.reshape(np.full(12, 0.8, dtype=np.float32), (3, 1, 4)), dtype=np.float32)
    minvals = np.array(-5.0, dtype=np.float32)
    maxvals = np.array(5.0, dtype=np.float32)
    name = "case4"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = [7, 1]
    seed = np.array([100, 200], dtype=np.int32)
    means = np.array([2.0], dtype=np.float32)
    stddevs = np.array(0.75, dtype=np.float32)
    minvals = np.array([0.0], dtype=np.float32)
    maxvals = np.array([3.0], dtype=np.float32)
    name = "case5"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (empty first dimension)
    shape = [0, 3]
    seed = np.array([77, 88], dtype=np.int32)
    means = np.array([0.0, -1.0, 2.0], dtype=np.float32)
    stddevs = np.array(1.2, dtype=np.float32)
    minvals = np.array(-1.0, dtype=np.float32)
    maxvals = np.array(1.0, dtype=np.float32)
    name = "case6"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (broadcast across 2 and 4)
    shape = [6, 2, 4]
    seed = np.array([555, 999], dtype=np.int64)
    means = np.array([[0.0], [1.5]], dtype=np.float64)
    stddevs = np.array([[0.4, 0.6, 0.8, 1.0]], dtype=np.float64)
    minvals = np.array([[-3.0], [-2.0]], dtype=np.float64)
    maxvals = np.array([[3.0, 2.5, 2.0, 1.5]], dtype=np.float64)
    name = "case7"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (full matrix parameters)
    shape = [3, 3]
    seed = np.array([42, 24], dtype=np.int32)
    means = np.array([[0.0, 1.0, 2.0],
                      [-1.0, -2.0, -3.0],
                      [0.5, -0.5, 1.5]], dtype=np.float32)
    stddevs = np.array([[0.5, 0.6, 0.7],
                        [1.0, 1.2, 0.8],
                        [0.3, 0.4, 0.9]], dtype=np.float32)
    minvals = np.array([[-2.0, -2.0, -2.0],
                        [-3.0, -4.0, -5.0],
                        [-1.0, -1.0, -1.0]], dtype=np.float32)
    maxvals = np.array([[2.0, 2.0, 2.0],
                        [3.0, 4.0, 5.0],
                        [1.0, 1.0, 1.0]], dtype=np.float32)
    name = "case8"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (negative mean window)
    shape = [1]
    seed = np.array([1, 2], dtype=np.int32)
    means = np.array(-5.0, dtype=np.float32)
    stddevs = np.array(0.1, dtype=np.float32)
    minvals = np.array(-5.2, dtype=np.float32)
    maxvals = np.array(-4.8, dtype=np.float32)
    name = "case9"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (float16 vector params)
    shape = [8, 5]
    seed = np.array([8, 16], dtype=np.int64)
    means = np.array([0.0, -1.0, 1.0, 2.0, -2.0], dtype=np.float16)
    stddevs = np.array([0.5, 0.25, 0.75, 1.0, 0.6], dtype=np.float16)
    minvals = np.array([-1.5, -2.0, -0.5, 0.0, -3.0], dtype=np.float16)
    maxvals = np.array([1.5, 0.0, 2.0, 3.0, -1.0 + 4.0], dtype=np.float16)
    name = "case10"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (mixed broadcast across 3,4,5)
    shape = [2, 3, 4, 5]
    seed = np.array([1010, 2020], dtype=np.int32)
    means = np.array(np.random.uniform(-1.0, 1.0, size=(3, 1, 5)).astype(np.float32))
    stddevs = np.array(np.full((1, 4, 1), 0.9, dtype=np.float32))
    minvals = np.array(-3.0, dtype=np.float32)
    maxvals = np.array(3.0, dtype=np.float32)
    name = "case11"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (vector params matching shape)
    shape = [9]
    seed = np.array([31415, 92653], dtype=np.int64)
    means = np.array(np.linspace(-2.0, 2.0, 9), dtype=np.float64)
    stddevs = np.array(np.linspace(0.2, 1.0, 9), dtype=np.float64)
    minvals = np.array(-4.0, dtype=np.float64)
    maxvals = np.array(4.0, dtype=np.float64)
    name = "case12"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_parameterized_truncated_normal_2"] = tf_random_stateless_parameterized_truncated_normal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_parameterized_truncated_normal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_parameterized_truncated_normal_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_parameterized_truncated_normal', generated_inputs['tf.random.stateless_parameterized_truncated_normal_2'], lib="tf", suffix=2)
