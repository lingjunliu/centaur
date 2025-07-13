
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_cross_inputs():
    list_of_inputs = []

    # Input 1: Basic RaggedTensors
    inputs = [tf.ragged.constant([['a'], ['b', 'c']]),
              tf.ragged.constant([['d'], ['e']]),
              tf.ragged.constant([['f'], ['g']])]
    name = None
    input_dict = {'inputs': inputs, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty RaggedTensor
    inputs = [tf.ragged.constant([['a'], ['b', 'c']]),
              tf.ragged.constant([[]]),
              tf.ragged.constant([['f'], ['g']])]
    name = 'empty_test'
    input_dict = {'inputs': inputs, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Dense Tensors
    inputs = [tf.constant([['a'], ['b']]),
              tf.constant([['c'], ['d']]),
              tf.constant([['e'], ['f']])]
    name = None
    input_dict = {'inputs': inputs, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed Ragged and Dense
    inputs = [tf.ragged.constant([['a'], ['b', 'c']]),
              tf.constant([['d'], ['e']]),
              tf.ragged.constant([['f'], ['g']])]
    name = 'mixed'
    input_dict = {'inputs': inputs, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: SparseTensor
    inputs = [tf.ragged.constant([['a'], ['b', 'c']]),
              tf.sparse.from_dense(tf.constant([['d'], ['e']])),
              tf.ragged.constant([['f'], ['g']])]
    name = 'sparse_test'
    input_dict = {'inputs': inputs, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All Empty Ragged Tensors
    inputs = [tf.ragged.constant([[]]),
              tf.ragged.constant([[]]),
              tf.ragged.constant([[]])]
    name = "all_empty"
    input_dict = {'inputs': inputs, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ragged.cross"] = tf_ragged_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ragged.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.cross'.")

check_valid('tf.ragged.cross', generated_inputs['tf.ragged.cross'], lib="tf", suffix=0)
