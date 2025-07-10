
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_accumulate_n_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integers
    inputs = [tf.constant([[1, 2], [3, 4]]), tf.constant([[5, 6], [7, 8]])]
    shape = [2, 2]
    tensor_dtype = tf.int32
    name = "accumulate_example_1"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.accumulate_n"] = tf_math_accumulate_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.accumulate_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.accumulate_n'.")

check_valid('tf.math.accumulate_n', generated_inputs['tf.math.accumulate_n'], lib="tf", suffix=0)
