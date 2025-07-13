
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_in_top_k_inputs():
    list_of_inputs = []

    # Input 1
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([2, 0], dtype=np.int32)
    k = 1
    name = "in_top_k_1"
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    predictions = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)
    targets = np.array([0, 1], dtype=np.int64)
    k = 2
    name = "in_top_k_2"
    input_dict = {"predictions": predictions, "targets": targets, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.InTopK"] = tf_raw_ops_in_top_k_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.InTopK' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.InTopK'.")

check_valid('tf.raw_ops.InTopK', generated_inputs['tf.raw_ops.InTopK'], lib="tf", suffix=0)
