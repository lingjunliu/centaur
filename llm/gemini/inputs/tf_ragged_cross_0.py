
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_ragged_cross_inputs():
    list_of_inputs = []

    # Input 1: Basic example with RaggedTensors
    inputs = [tf.ragged.constant([['a'], ['b', 'c']]),
              tf.ragged.constant([['d'], ['e']]),
              tf.ragged.constant([['f'], ['g']])]
    name = None
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(input_dict)

    # Input 2: Using Tensors
    inputs = [tf.constant([['a'], ['b']]),
              tf.constant([['c'], ['d']])]
    name = "my_cross"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(input_dict)

    # Input 3: Mixed inputs (RaggedTensor and Tensor)
    inputs = [tf.ragged.constant([['a'], ['b']]),
              tf.constant([['c'], ['d']])]
    name = None
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(input_dict)

    # Input 4: More complex RaggedTensor structure
    inputs = [tf.ragged.constant([['a', 'b'], ['c']]),
              tf.ragged.constant([['d'], ['e', 'f']])]
    name = None
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(input_dict)

    # Input 5: Empty RaggedTensor
    inputs = [tf.ragged.constant([[], []]),
              tf.constant([['a'], ['b']])]
    name = None
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.ragged.cross"] = tf_ragged_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.ragged.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.cross'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.ragged.cross', generated_inputs['tf.ragged.cross'], lib="tf", suffix=0)
