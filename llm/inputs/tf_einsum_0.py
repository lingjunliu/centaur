
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_einsum_inputs():
    list_of_inputs = []

    # Input 1: Matrix multiplication
    m0 = np.random.normal(size=[2, 3]).astype(np.float32)
    m1 = np.random.normal(size=[3, 5]).astype(np.float32)
    equation = 'ij,jk->ik'
    inputs = [m0, m1]
    optimize = 'greedy'
    name = 'matrix_mult'
    input_dict = {'equation': equation, 'inputs': inputs, 'optimize': optimize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Dot product
    u = np.random.normal(size=[5]).astype(np.float32)
    v = np.random.normal(size=[5]).astype(np.float32)
    equation = 'i,i->'
    inputs = [u, v]
    optimize = 'optimal'
    name = 'dot_product'
    input_dict = {'equation': equation, 'inputs': inputs, 'optimize': optimize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Outer product
    u = np.random.normal(size=[3]).astype(np.float32)
    v = np.random.normal(size=[5]).astype(np.float32)
    equation = 'i,j->ij'
    inputs = [u, v]
    optimize = 'branch-2'
    name = 'outer_product'
    input_dict = {'equation': equation, 'inputs': inputs, 'optimize': optimize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Transpose
    m = np.ones((2, 3)).astype(np.float32)
    equation = 'ij->ji'
    inputs = [m]
    optimize = 'branch-all'
    name = 'transpose'
    input_dict = {'equation': equation, 'inputs': inputs, 'optimize': optimize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Trace
    m = np.reshape(np.arange(9), [3, 3]).astype(np.float32)
    equation = 'ii'
    inputs = [m]
    optimize = 'auto'
    name = 'trace'
    input_dict = {'equation': equation, 'inputs': inputs, 'optimize': optimize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.einsum"] = tf_einsum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.einsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.einsum'.")

check_valid('tf.einsum', generated_inputs['tf.einsum'], lib="tf", suffix=0)
