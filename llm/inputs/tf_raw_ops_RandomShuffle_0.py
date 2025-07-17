
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_RandomShuffle_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D tensor, with seed
    value = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    seed = 1
    seed2 = 1
    name = None
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different seed
    value = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    seed = 123
    seed2 = 456
    name = "shuffle_with_seed"
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float tensor, with seed
    value = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], dtype=np.float32)
    seed = 2
    seed2 = 2
    name = None
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor, with seed
    value = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    seed = 3
    seed2 = 3
    name = None
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor, with seed
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    seed = 4
    seed2 = 4
    name = None
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty tensor, with seed
    value = np.array([], dtype=np.int32).reshape(0,1)
    seed = 5
    seed2 = 5
    name = None
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bool Tensor, with seed
    value = np.array([[True, False], [False, True], [True, True]], dtype=np.bool_)
    seed = 7
    seed2 = 7
    name = None
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative Values, with seed
    value = np.array([[-1, -2], [-3, -4], [-5, -6]], dtype=np.int32)
    seed = 8
    seed2 = 8
    name = None
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Higher rank, different shape, with seed
    value = np.random.rand(2, 3, 4).astype(np.float32)
    seed = 9
    seed2 = 9
    name = "high_rank_shuffle"
    input_dict = {"value": value, "seed": seed, "seed2": seed2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int64
    value = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    seed = 10
    seed2 = 10
    name = None
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
