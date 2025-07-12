
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_einsum_inputs():
    list_of_inputs = []

    # Input 1: Basic dot product
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    subscripts = "ij,jk->ik"
    operands = [a, b]

    input_dict = {
        "subscripts": subscripts,
        "operands": [tf.convert_to_tensor(op) for op in operands]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Trace
    a = np.array([[1, 2], [3, 4]])
    subscripts = "ii->"
    operands = [a]

    input_dict = {
        "subscripts": subscripts,
        "operands": [tf.convert_to_tensor(op) for op in operands]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Transpose
    a = np.array([[1, 2], [3, 4]])
    subscripts = "ij->ji"
    operands = [a]
    input_dict = {
        "subscripts": subscripts,
        "operands": [tf.convert_to_tensor(op) for op in operands]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Sum
    a = np.array([[1, 2], [3, 4]])
    subscripts = "ij->"
    operands = [a]

    input_dict = {
        "subscripts": subscripts,
        "operands": [tf.convert_to_tensor(op) for op in operands]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Matrix-vector multiplication
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    subscripts = "ij,j->i"
    operands = [a, b]

    input_dict = {
        "subscripts": subscripts,
        "operands": [tf.convert_to_tensor(op) for op in operands]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batch matrix multiplication
    a = np.random.rand(2, 3, 4).astype(np.float32)
    b = np.random.rand(2, 4, 5).astype(np.float32)
    subscripts = "ijk,ikl->ijl"
    operands = [a, b]
    input_dict = {
        "subscripts": subscripts,
        "operands": [tf.convert_to_tensor(op) for op in operands]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Outer product
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    subscripts = "i,j->ij"
    operands = [a, b]

    input_dict = {
        "subscripts": subscripts,
        "operands": [tf.convert_to_tensor(op) for op in operands]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: More complex contraction
    a = np.random.rand(3, 4, 5).astype(np.float32)
    b = np.random.rand(4, 6).astype(np.float32)
    c = np.random.rand(5, 6, 7).astype(np.float32)
    subscripts = "ijk,jl,klm->im"
    operands = [a, b, c]
    input_dict = {
        "subscripts": subscripts,
        "operands": [tf.convert_to_tensor(op) for op in operands]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Repeated indices
    a = np.array([[1, 2], [3, 4]])
    subscripts = "ii->"
    operands = [a]
    input_dict = {
        "subscripts": subscripts,
        "operands": [tf.convert_to_tensor(op) for op in operands]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10:
    a = np.array([1,2,3])
    b = np.array([4,5,6])
    subscripts = "i,i->"
    operands = [a,b]
    input_dict = {
        "subscripts": subscripts,
        "operands": [tf.convert_to_tensor(op) for op in operands]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.einsum"] = tf_experimental_numpy_einsum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.einsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.einsum'.")

check_valid('tf.experimental.numpy.einsum', generated_inputs['tf.experimental.numpy.einsum'], lib="tf", suffix=0)
