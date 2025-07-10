
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
    inputs = [tf.constant(m0), tf.constant(m1)]
    input_dict = {'equation': equation, 'inputs': inputs.copy(), 'optimize': 'greedy', 'name': 'matmul'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Dot product
    u = np.random.normal(size=[5]).astype(np.float32)
    v = np.random.normal(size=[5]).astype(np.float32)
    equation = 'i,i->'
    inputs = [tf.constant(u), tf.constant(v)]
    input_dict = {'equation': equation, 'inputs': inputs.copy(), 'optimize': 'optimal', 'name': 'dot'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Outer product
    u = np.random.normal(size=[3]).astype(np.float32)
    v = np.random.normal(size=[5]).astype(np.float32)
    equation = 'i,j->ij'
    inputs = [tf.constant(u), tf.constant(v)]
    input_dict = {'equation': equation, 'inputs': inputs.copy(), 'optimize': 'branch-2', 'name': 'outer'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Transpose
    m = np.random.normal(size=[2, 3]).astype(np.float32)
    equation = 'ij->ji'
    inputs = [tf.constant(m)]
    input_dict = {'equation': equation, 'inputs': inputs.copy(), 'optimize': 'branch-all', 'name': 'transpose'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Trace
    m = np.reshape(np.arange(9), [3, 3]).astype(np.float32)
    equation = 'ii->'
    inputs = [tf.constant(m)]
    input_dict = {'equation': equation, 'inputs': inputs.copy(), 'optimize': 'auto', 'name': 'trace'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batch matrix multiplication
    s = np.random.normal(size=[7, 5, 3]).astype(np.float32)
    t = np.random.normal(size=[7, 3, 2]).astype(np.float32)
    equation = 'bij,bjk->bik'
    inputs = [tf.constant(s), tf.constant(t)]
    input_dict = {'equation': equation, 'inputs': inputs.copy(), 'optimize': 'greedy', 'name': 'batch_matmul'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting batch matrix multiplication
    s = np.random.normal(size=[11, 1, 5, 3]).astype(np.float32)
    t = np.random.normal(size=[1, 7, 3, 2]).astype(np.float32)
    equation = '...ij,...jk->...ik'
    inputs = [tf.constant(s), tf.constant(t)]
    input_dict = {'equation': equation, 'inputs': inputs.copy(), 'optimize': 'optimal', 'name': 'broadcast_matmul'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: More complex contraction
    a = np.random.normal(size=[3, 4, 5]).astype(np.float32)
    b = np.random.normal(size=[4, 6]).astype(np.float32)
    equation = 'ijk,jl->ilk'
    inputs = [tf.constant(a), tf.constant(b)]
    input_dict = {'equation': equation, 'inputs': inputs.copy(), 'optimize': 'branch-2', 'name': 'complex_contract'}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: Summation over all axes
    a = np.random.normal(size=[2, 3, 4]).astype(np.float32)
    equation = 'ijk->'
    inputs = [tf.constant(a)]
    input_dict = {'equation': equation, 'inputs': inputs.copy(), 'optimize': 'branch-all', 'name': 'sum_all'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Diag
    m = np.reshape(np.arange(16), [4,4]).astype(np.float32)
    equation = 'ii->i'
    inputs = [tf.constant(m)]
    input_dict = {'equation': equation, 'inputs': inputs.copy(), 'optimize': 'auto', 'name': 'diag'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.einsum"] = tf_einsum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.einsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.einsum'.")

check_valid('tf.einsum', generated_inputs['tf.einsum'], lib="tf", suffix=0)
