
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_random_stateless_gamma_inputs():
    list_of_inputs = []

    shape = np.array([10, 2], dtype=np.int32)
    seed = np.array([12, 34], dtype=np.int64)
    alpha = np.array([0.5, 1.5], dtype=np.float32)
    beta = np.array(1.0, dtype=np.float32)
    dtype = np.float32
    name = "case1_basic_vec"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([7, 5, 2], dtype=np.int32)
    seed = np.array([123, 456], dtype=np.int32)
    alpha = np.array([0.5, 1.5], dtype=np.float64)
    beta = np.array(2.0, dtype=np.float64)
    dtype = np.float64
    name = "case2_broadcast_lastdim"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([30, 3, 2], dtype=np.int32)
    seed = np.array([98765, 43210], dtype=np.int64)
    alpha = np.array([[1.0], [3.0], [5.0]], dtype=np.float32)
    beta = np.array([[3.0, 4.0]], dtype=np.float32)
    dtype = np.float32
    name = "case3_matrix_broadcast"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([100], dtype=np.int32)
    seed = np.array([1, 0], dtype=np.int32)
    alpha = np.array(2.0, dtype=np.float64)
    beta = np.array(0.5, dtype=np.float64)
    dtype = np.float64
    name = "case4_scalar_params"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([6, 8, 4, 3], dtype=np.int32)
    seed = np.array([314159, 265358], dtype=np.int64)
    alpha = np.linspace(0.1, 2.5, num=8 * 4 * 3, dtype=np.float32).reshape(8, 4, 3)
    beta = np.array(1.0, dtype=np.float32)
    dtype = np.float32
    name = "case5_highrank_alpha"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([4, 3, 2, 5], dtype=np.int32)
    seed = np.array([42, 99], dtype=np.int32)
    alpha = np.array([[1.0, 2.0, 1.5, 0.7, 3.0],
                      [2.5, 1.2, 0.8, 2.2, 1.1]], dtype=np.float32)
    beta = np.array([[1.0, 2.0, 1.0, 0.5, 3.0]], dtype=np.float32)
    dtype = np.float32
    name = "case6_partial_broadcast"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([2, 3, 4], dtype=np.int32)
    seed = np.array([7, 13], dtype=np.int64)
    alpha = np.array([[0.8], [1.2], [2.5]], dtype=np.float32)
    beta = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    dtype = np.float32
    name = "case7_mixed_rank_broadcast"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([1000, 1], dtype=np.int64)
    seed = np.array([0, 1], dtype=np.int32)
    alpha = np.array([0.2], dtype=np.float32)
    beta = np.array([50.0], dtype=np.float32)
    dtype = np.float32
    name = "case8_small_alpha_large_beta"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([5, 2], dtype=np.int32)
    seed = np.array([2147483646, 2147483645], dtype=np.int64)
    alpha = np.array([2.0, 3.0], dtype=np.float16)
    beta = np.array([0.5, 2.0], dtype=np.float16)
    dtype = np.float16
    name = "case9_float16_dtype"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([2, 2, 2], dtype=np.int32)
    seed = np.array([888, 777], dtype=np.int32)
    alpha = np.array([[1.0, 2.0],
                      [3.0, 4.0]], dtype=np.float32)
    beta = np.array([[1.0],
                     [0.5]], dtype=np.float32)
    dtype = np.float32
    name = "case10_beta_needs_broadcast"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([9, 3], dtype=np.int64)
    seed = np.array([2025, 1108], dtype=np.int64)
    alpha = np.array([0.7, 1.3, 2.1], dtype=np.float64)
    beta = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    dtype = np.float64
    name = "case11_int64_shape_and_seed"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([3, 4, 1], dtype=np.int32)
    seed = np.array([12, 34], dtype=np.int32)
    alpha = np.array([[0.1],
                      [0.2],
                      [0.3],
                      [0.4]], dtype=np.float32)
    beta = np.array(10.0, dtype=np.float32)
    dtype = np.float32
    name = "case12_trailing_two_dims"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_gamma_1"] = tf_random_stateless_gamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_gamma_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_gamma_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_gamma', generated_inputs['tf.random.stateless_gamma_1'], lib="tf", suffix=1)
