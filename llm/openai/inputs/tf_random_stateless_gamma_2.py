
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_gamma_inputs():
    list_of_inputs = []

    # Input 1
    shape = [np.int32(10), np.int32(2)]
    seed = [np.int32(12), np.int32(34)]
    alpha = [np.float32(0.5), np.float32(1.5)]
    beta = [np.float32(1.0), np.float32(1.0)]
    dtype = np.float32
    name = "gamma_case_1"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = [np.int64(7), np.int64(5), np.int64(2)]
    seed = [np.int64(123), np.int64(456)]
    alpha = [np.float32(0.5), np.float32(1.5)]
    beta = [np.float32(1.0)]
    dtype = np.float32
    name = "gamma_case_2"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = [np.int32(30), np.int32(3), np.int32(2)]
    seed = [np.int32(12), np.int32(34)]
    alpha = [[np.float64(1.0)], [np.float64(3.0)], [np.float64(5.0)]]
    beta = [[np.float64(3.0), np.float64(4.0)]]
    dtype = np.float64
    name = "gamma_case_3"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = [np.int32(4), np.int32(3), np.int32(1)]
    seed = [np.int32(9876), np.int32(5432)]
    alpha = [[np.float32(0.7)], [np.float32(2.3)], [np.float32(5.1)]]
    beta = [np.float32(2.0)]
    dtype = np.float32
    name = "gamma_case_4"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = [np.int32(5), np.int32(2), np.int32(1), np.int32(3)]
    seed = [np.int32(42), np.int32(24)]
    alpha = [
        [[np.float32(0.5), np.float32(1.0), np.float32(1.5)]],
        [[np.float32(2.0), np.float32(2.5), np.float32(3.0)]]
    ]  # shape (2,1,3)
    beta = [np.float32(0.8), np.float32(1.2), np.float32(2.0)]  # shape (3,)
    dtype = np.float32
    name = "gamma_case_5"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = [np.int32(8), np.int32(1)]
    seed = [np.int32(7), np.int32(11)]
    alpha = [np.float32(2.0)]
    beta = [np.float32(0.5)]
    dtype = np.float32
    name = "gamma_case_6"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = [np.int32(2), np.int32(4), np.int32(2)]
    seed = [np.int32(111), np.int32(222)]
    alpha = [np.float16(0.2), np.float16(5.0)]
    beta = [np.float16(10.0), np.float16(0.8)]
    dtype = np.float16
    name = "gamma_case_7"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = [np.int64(4), np.int64(3), np.int64(2)]
    seed = [np.int64(13579), np.int64(24680)]
    alpha = [
        [np.float64(1.1), np.float64(2.2)],
        [np.float64(3.3), np.float64(4.4)],
        [np.float64(5.5), np.float64(6.6)]
    ]  # shape (3,2)
    beta = [
        [np.float64(0.5)],
        [np.float64(1.5)],
        [np.float64(2.5)]
    ]  # shape (3,1)
    dtype = np.float64
    name = "gamma_case_8"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = [np.int32(2), np.int32(2)]
    seed = [np.int32(333), np.int32(444)]
    alpha = [
        [np.float32(0.9)],
        [np.float32(1.8)]
    ]  # shape (2,1)
    beta = [
        [np.float32(1.0), np.float32(2.0)]
    ]  # shape (1,2)
    dtype = np.float32
    name = "gamma_case_9"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = [np.int32(2), np.int32(2), np.int32(2), np.int32(1)]
    seed = [np.int32(555), np.int32(666)]
    alpha = [np.float32(3.0)]
    beta = [np.float32(4.0)]
    dtype = np.float32
    name = "gamma_case_10"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    shape = [np.int32(3), np.int32(1)]
    seed = [np.int32(777), np.int32(888)]
    alpha = [np.float64(0.05)]
    beta = [np.float64(50.0)]
    dtype = np.float64
    name = "gamma_case_11"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    shape = [np.int32(9), np.int32(4)]
    seed = [np.int32(1010), np.int32(2020)]
    alpha = [np.float32(1.0), np.float32(2.0), np.float32(3.0), np.float32(4.0)]  # shape (4,)
    beta = [np.float32(0.5), np.float32(1.5), np.float32(2.5), np.float32(3.5)]  # shape (4,)
    dtype = np.float32
    name = "gamma_case_12"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_gamma_2"] = tf_random_stateless_gamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_gamma_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_gamma_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_gamma', generated_inputs['tf.random.stateless_gamma_2'], lib="tf", suffix=2)
