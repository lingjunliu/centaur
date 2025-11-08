
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_identity_n_inputs():
    list_of_inputs = []
    input_1 = [tf.constant(np.array([1, 2, 3], dtype=np.int32)), tf.constant(np.array([4.5, 5.5, 6.5], dtype=np.float32))]
    input_1 = [input_1[0], input_1[1]]

    list_of_inputs.append({"input": input_1, "name": "input_1"})

    return list_of_inputs

generated_inputs["tf.identity_n"] = tf_identity_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.identity_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.identity_n'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.identity_n', generated_inputs['tf.identity_n'], lib="tf", suffix=0)
