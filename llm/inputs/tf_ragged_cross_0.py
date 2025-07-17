
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_cross_inputs():
    list_of_inputs = []

    # Input 1: Basic example with RaggedTensors
    inputs = [np.array([['a'], ['b', 'c']], dtype=np.str_),
              np.array([['d'], ['e']], dtype=np.str_),
              np.array([['f'], ['g']], dtype=np.str_)]
    name = None
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Using Tensors instead of RaggedTensors
    inputs = [np.array([['a'], ['b']], dtype=np.str_),
              np.array([['c'], ['d']], dtype=np.str_)]
    name = "cross_product"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixing Tensors and RaggedTensors
    inputs = [np.array([['a'], ['b', 'c']], dtype=np.str_),
              np.array([['d'], ['e']], dtype=np.str_)]
    name = None
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With different string lengths
    inputs = [np.array([['long_string'], ['short']], dtype=np.str_),
              np.array([['another'], ['yet_another']], dtype=np.str_)]
    name = None
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Using SparseTensors
    inputs = [np.array([['a'], ['b']], dtype=np.str_),
              np.array([['c'], ['d']], dtype=np.str_)]
    name = None
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: More SparseTensors combined with RaggedTensor
    inputs = [np.array([['a'], ['b']], dtype=np.str_),
              np.array([['c'], ['d', 'e']], dtype=np.str_)]
    name = "sparse_cross"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Three inputs with varying row lengths
    inputs = [np.array([['a'], ['b', 'c']], dtype=np.str_),
              np.array([['d'], ['e']], dtype=np.str_),
              np.array([['f'], ['g']], dtype=np.str_)]
    name = None
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty strings
    inputs = [np.array([[''], ['b']], dtype=np.str_),
              np.array([['c'], ['']], dtype=np.str_)]
    name = None
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Mixed empty strings and valid strings
    inputs = [np.array([['a'], ['']], dtype=np.str_),
              np.array([[''], ['b']], dtype=np.str_)]
    name = None
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger ragged tensor
    inputs = [np.array([['a', 'b'], ['c', 'd', 'e']], dtype=np.str_),
              np.array([['f'], ['g']], dtype=np.str_)]
    name = None
    input_dict = {"inputs": inputs, "name": name}
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
