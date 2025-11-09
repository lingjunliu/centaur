
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_randomshuffle_inputs():
    list_of_inputs = []

    # Input 1: 1D int32
    value = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    seed = np.int32(1)
    seed2 = np.int32(2)
    name = "shuffle_1d_int32"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 2: 2D float32 with negatives
    value = np.array([[1.5, -2.3], [3.0, 4.1], [0.0, 9.9]], dtype=np.float32)
    seed = np.int64(42)
    seed2 = np.int64(7)
    name = "shuffle_2d_float32"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 3: 2D int64 with negatives
    value = np.array([[-1, -2, -3], [4, 5, 6]], dtype=np.int64)
    seed = np.int64(7)
    seed2 = np.int64(99)
    name = "shuffle_2d_int64_neg"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 4: 2D bool
    value = np.array([[True, False], [False, True], [True, True], [False, False]], dtype=np.bool_)
    seed = np.int32(123)
    seed2 = np.int32(456)
    name = "shuffle_2d_bool"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 5: 3D float16
    value = (np.arange(2 * 3 * 4, dtype=np.float16).reshape(2, 3, 4) - np.float16(10.5))
    seed = np.int32(11)
    seed2 = np.int32(22)
    name = "shuffle_3d_float16"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 6: 2D float64 single row
    value = np.array([[-1.0, -2.0, 0.0, 2.5, 10.75]], dtype=np.float64)
    seed = np.int32(1)
    seed2 = np.int32(3)
    name = "shuffle_2d_float64_row"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 7: 2D uint8
    value = np.arange(24, dtype=np.uint8).reshape(6, 4)
    seed = np.int32(2024)
    seed2 = np.int32(8)
    name = "shuffle_2d_uint8"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 8: 4D int16
    value = np.random.randint(-100, 100, size=(5, 2, 3, 4)).astype(np.int16)
    seed = np.int64(8080)
    seed2 = np.int64(9090)
    name = "shuffle_4d_int16"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 9: 5D float32
    value = np.arange(12, dtype=np.float32).reshape(2, 1, 3, 1, 2)
    seed = np.int64(314159)
    seed2 = np.int64(2653)
    name = "shuffle_5d_float32"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 10: Empty along first dim, float64
    value = np.zeros((0, 5), dtype=np.float64)
    seed = np.int64(100)
    seed2 = np.int64(200)
    name = "shuffle_empty_first_dim"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    return list_of_inputs

generated_inputs["tf.raw_ops.RandomShuffle"] = tf_raw_ops_randomshuffle_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RandomShuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomShuffle'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.RandomShuffle', generated_inputs['tf.raw_ops.RandomShuffle'], lib="tf", suffix=0)
