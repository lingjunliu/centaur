
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_random_stateless_parameterized_truncated_normal_inputs():
    list_of_inputs = []

    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([7, 17], dtype=np.int32)
    means = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    stddevs = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    minvals = np.array([-2.0, -1.0, -3.0], dtype=np.float32)
    maxvals = np.array([2.0, 3.0, 1.0], dtype=np.float32)
    name = "test1"
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_parameterized_truncated_normal"] = tf_random_stateless_parameterized_truncated_normal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_parameterized_truncated_normal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_parameterized_truncated_normal'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_parameterized_truncated_normal', generated_inputs['tf.random.stateless_parameterized_truncated_normal'], lib="tf", suffix=0)
