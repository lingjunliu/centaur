
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_random_stateless_uniform_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    minval = np.array(0.0, dtype=np.float32)
    maxval = np.array(1.0, dtype=np.float32)
    dtype = tf.float32
    name = "uniform1"
    alg = "auto_select"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "minval": minval,
        "maxval": maxval,
        "dtype": dtype,
        "name": name,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32
    shape = np.array([4, 5], dtype=np.int32)
    seed = np.array([3, 4], dtype=np.int32)
    minval = np.array(0, dtype=np.int32)
    maxval = np.array(10, dtype=np.int32)
    dtype = tf.int32
    name = "uniform2"
    alg = "philox"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "minval": minval,
        "maxval": maxval,
        "dtype": dtype,
        "name": name,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, different shape
    shape = np.array([10], dtype=np.int32)
    seed = np.array([5, 6], dtype=np.int32)
    minval = np.array(-1.0, dtype=np.float64)
    maxval = np.array(1.0, dtype=np.float64)
    dtype = tf.float64
    name = "uniform3"
    alg = "threefry"

    input_dict = {
        "shape": shape,
        "seed": seed,
        "minval": minval,
        "maxval": maxval,
        "dtype": dtype,
        "name": name,
        "alg": alg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.stateless_uniform"] = tf_random_stateless_uniform_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_uniform' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_uniform'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_uniform', generated_inputs['tf.random.stateless_uniform'], lib="tf", suffix=0)
