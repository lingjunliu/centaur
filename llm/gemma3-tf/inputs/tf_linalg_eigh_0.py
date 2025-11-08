
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_eigh_inputs():
    list_of_inputs = []

    input_dict = {
        "tensor": np.array([[[1.0, 2.0], [2.0, 1.0]]], dtype=np.float64),
        "name": "eigh_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        "name": "eigh_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.eigh"] = tf_linalg_eigh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.eigh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.eigh'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.eigh', generated_inputs['tf.linalg.eigh'], lib="tf", suffix=0)
