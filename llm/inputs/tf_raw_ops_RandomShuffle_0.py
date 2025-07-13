
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_RandomShuffle_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    value = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    seed = 0
    seed2 = 0
    name = "shuffle_1"
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor
    value = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    seed = 1
    seed2 = 2
    name = "shuffle_2"
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    value = np.random.rand(3, 2, 2).astype(np.float64)
    seed = 123
    seed2 = 456
    name = "shuffle_3"
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with negative values
    value = np.array([-1, -2, 3, -4, 5], dtype=np.int64)
    seed = 7
    seed2 = 8
    name = "shuffle_4"
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with zero values
    value = np.array([0, 0, 1, 0, 2], dtype=np.int32)
    seed = 9
    seed2 = 10
    name = "shuffle_5"
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large seed values, but still within the int32 range (TF seems to convert seed to int32)
    value = np.array([1, 2, 3], dtype=np.int32)
    seed = 2147483647
    seed2 = 2147483646
    name = "shuffle_6"
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty name
    value = np.array([1, 2], dtype=np.float32)
    seed = 11
    seed2 = 12
    name = ""
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different dtype
    value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    seed = 13
    seed2 = 14
    name = "shuffle_8"
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Higher rank tensor (4D)
    value = np.random.rand(2, 2, 2, 2).astype(np.float32)
    seed = 15
    seed2 = 16
    name = "shuffle_9"
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different shape
    value = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    seed = 17
    seed2 = 18
    name = "shuffle_10"
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RandomShuffle"] = tf_raw_ops_RandomShuffle_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RandomShuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomShuffle'.")

check_valid('tf.raw_ops.RandomShuffle', generated_inputs['tf.raw_ops.RandomShuffle'], lib="tf", suffix=0)
