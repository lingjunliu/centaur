
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_randomshuffle_inputs():
    list_of_inputs = []

    input_dict = {
        'value': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        'seed': 42,
        'seed2': 100,
        'name': 'shuffle_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'value': np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32),
        'seed': 42,
        'seed2': 0,
        'name': 'shuffle_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'value': np.array([1, 2, 3, 4, 5], dtype=np.int64),
        'seed': 42,
        'seed2': 5,
        'name': 'shuffle_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
