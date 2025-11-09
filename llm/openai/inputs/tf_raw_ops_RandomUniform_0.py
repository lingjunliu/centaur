
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_randomuniform_inputs():
    list_of_inputs = []

    shape = np.array([2, 3], dtype=np.int32)
    dtype = np.float32
    seed = 7
    seed2 = 11
    name = "ru_2x3_f32"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([], dtype=np.int32)
    dtype = np.float64
    seed = 123
    seed2 = 1
    name = "ru_scalar_f64"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([5], dtype=np.int64)
    dtype = np.float16
    seed = 456
    seed2 = 789
    name = "ru_len5_f16"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([2, 0, 4], dtype=np.int32)
    dtype = np.float32
    seed = 999
    seed2 = 1001
    name = "ru_zero_dim_mid_f32"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([1, 1, 1], dtype=np.int64)
    dtype = np.float64
    seed = 42
    seed2 = 24
    name = "ru_ones_f64"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([10, 10], dtype=np.int32)
    dtype = np.float16
    seed = 2021
    seed2 = 2022
    name = "ru_10x10_f16_bothseed"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([3, 4, 5, 6], dtype=np.int64)
    dtype = np.float32
    seed = 31415
    seed2 = 27182
    name = "ru_3x4x5x6_f32"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([7], dtype=np.int32)
    dtype = np.float64
    seed = 707
    seed2 = 808
    name = "ru_len7_f64"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([2, 3, 1, 0, 4], dtype=np.int64)
    dtype = np.float32
    seed = 8080
    seed2 = 9090
    name = "ru_5d_with_zero_f32"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([1000], dtype=np.int32)
    dtype = np.float32
    seed = 1
    seed2 = 2
    name = "ru_len1000_f32"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([8, 8, 8], dtype=np.int32)
    dtype = np.float64
    seed = 123456
    seed2 = 654321
    name = "ru_8cube_f64"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([0], dtype=np.int64)
    dtype = np.float16
    seed = 42
    seed2 = 43
    name = "ru_emptyvec_f16"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    return list_of_inputs

generated_inputs["tf.raw_ops.RandomUniform"] = tf_raw_ops_randomuniform_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RandomUniform' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomUniform'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.RandomUniform', generated_inputs['tf.raw_ops.RandomUniform'], lib="tf", suffix=0)
