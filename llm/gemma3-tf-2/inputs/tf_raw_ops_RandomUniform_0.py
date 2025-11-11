
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_random_uniform_inputs():
    list_of_inputs = []

    input_dict = {
        "shape": np.array([2, 3], dtype=np.int32),
        "dtype": np.float32,
        "seed": 10,
        "seed2": 20,
        "name": "test_uniform_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "shape": np.array([4], dtype=np.int64),
        "dtype": np.float64,
        "seed": 123,
        "seed2": 456,
        "name": "test_uniform_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.RandomUniform"] = tf_raw_ops_random_uniform_inputs()

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
