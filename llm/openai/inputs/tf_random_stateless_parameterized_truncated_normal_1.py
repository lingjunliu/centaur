
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_random_stateless_parameterized_truncated_normal_inputs():
    list_of_inputs = []

    shape = np.array([10], dtype=np.int32)
    seed = np.array([123, 456], dtype=np.int32)
    means = np.array(0.0, dtype=np.float32)
    stddevs = np.array(1.0, dtype=np.float32)
    minvals = np.array(-2.0, dtype=np.float32)
    maxvals = np.array(2.0, dtype=np.float32)
    name = "basic_float32_scalar_params"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([5, 4], dtype=np.int64)
    seed = np.array([7, 17], dtype=np.int64)
    means = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    stddevs = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    minvals = np.array([-1.0, -1.0, -2.0, -3.0], dtype=np.float32)
    maxvals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    name = "per_column_params_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([3, 2, 3], dtype=np.int32)
    seed = np.array([42, 24], dtype=np.int32)
    means = np.array([[0.0, 0.5, -0.5], [1.0, -1.0, 2.0]], dtype=np.float32)
    stddevs = np.array([[1.0, 0.2, 0.3], [0.5, 2.0, 1.5]], dtype=np.float32)
    minvals = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    maxvals = np.array([[3.0], [1.0]], dtype=np.float32)
    name = "broadcasting_example_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([5, 2, 1], dtype=np.int64)
    seed = np.array([3, 999999], dtype=np.int64)
    means = np.array(-0.5, dtype=np.float64)
    stddevs = np.array([[0.1], [2.0]], dtype=np.float64)
    minvals = np.array(-1.5, dtype=np.float64)
    maxvals = np.array(1.5, dtype=np.float64)
    name = "float64_mixed_rank_params"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([4, 1, 3, 1], dtype=np.int32)
    seed = np.array([31415, 27182], dtype=np.int32)
    means = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32).reshape(1, 3, 1)
    stddevs = np.array([0.5, 1.0, 1.5], dtype=np.float32).reshape(1, 3, 1)
    minvals = np.array(-0.75, dtype=np.float32)
    maxvals = np.array(0.75, dtype=np.float32)
    name = "higher_rank_broadcast_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([0, 7], dtype=np.int32)
    seed = np.array([2021, 2022], dtype=np.int64)
    means = np.array([0.0, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6], dtype=np.float32)
    stddevs = np.array([0.3] * 7, dtype=np.float32)
    minvals = np.array([-0.4] * 7, dtype=np.float32)
    maxvals = np.array([0.4] * 7, dtype=np.float32)
    name = "zero_sized_dim_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([6, 5, 4], dtype=np.int64)
    seed = np.array([1, 2], dtype=np.int32)
    means = np.array(0.0, dtype=np.float32)
    stddevs = np.array(0.5, dtype=np.float32)
    minvals = np.array(-1.0, dtype=np.float32)
    maxvals = np.array(1.0, dtype=np.float32)
    name = "float32_scalar_params_large_shape"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([8], dtype=np.int32)
    seed = np.array([123456789, 987654321], dtype=np.int64)
    means = np.linspace(-2.0, 2.0, 8, dtype=np.float32)
    stddevs = np.linspace(0.1, 1.0, 8, dtype=np.float32)
    minvals = np.full((8,), -5.0, dtype=np.float32)
    maxvals = np.full((8,), 5.0, dtype=np.float32)
    name = "vectorized_params_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([2, 3, 1, 5], dtype=np.int32)
    seed = np.array([0, 1], dtype=np.int32)
    means = (np.arange(3 * 1 * 5, dtype=np.float32).reshape(3, 1, 5) - 7.0) / 3.0
    stddevs = np.array(1.25, dtype=np.float32)
    minvals = np.full((1, 5), -0.5, dtype=np.float32)
    maxvals = np.full((3, 1, 5), 2.0, dtype=np.float32)
    name = "complex_broadcasting_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([2, 2], dtype=np.int64)
    seed = np.array([1234, 5678], dtype=np.int64)
    means = np.array([[-1.0, 1.0], [2.0, -2.0]], dtype=np.float64)
    stddevs = np.array([[0.1, 0.2], [3.0, 4.0]], dtype=np.float64)
    minvals = np.array(-0.2, dtype=np.float64)
    maxvals = np.array(0.2, dtype=np.float64)
    name = "float64_matrix_params"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([3], dtype=np.int32)
    seed = np.array([13579, 24680], dtype=np.int32)
    means = np.array([0.0, -0.5, 0.5], dtype=np.float32)
    stddevs = np.array([1e-6, 1e-3, 1e-2], dtype=np.float32)
    minvals = np.array(-1e-3, dtype=np.float32)
    maxvals = np.array(1e-3, dtype=np.float32)
    name = "tiny_stddevs_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([1, 2, 1, 3], dtype=np.int64)
    seed = np.array([101, 202], dtype=np.int64)
    means = np.array([[[0.0, 1.0, 2.0]], [[-1.0, -2.0, -3.0]]], dtype=np.float32)
    stddevs = np.array(0.7, dtype=np.float32)
    minvals = np.array([-0.5, -1.0, -1.5], dtype=np.float32)
    maxvals = np.array([[[0.5, 1.5, 2.5]], [[1.0, 2.0, 3.0]]], dtype=np.float32)
    name = "mixed_rank_tail_broadcast_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_parameterized_truncated_normal_1"] = tf_random_stateless_parameterized_truncated_normal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_parameterized_truncated_normal_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_parameterized_truncated_normal_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_parameterized_truncated_normal', generated_inputs['tf.random.stateless_parameterized_truncated_normal_1'], lib="tf", suffix=1)
