
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_einsum_inputs():
    list_of_inputs = []

    # Input 1
    subscripts = "ij,jk->ik"
    operands = [np.random.rand(2, 3).astype(np.float32), np.random.rand(3, 4).astype(np.float32)]
    input_dict = {"subscripts": subscripts, "operands": (tf.constant(operands[0]), tf.constant(operands[1]))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    subscripts = "i->i"
    operands = [np.random.rand(5).astype(np.float32)]
    input_dict = {"subscripts": subscripts, "operands": (tf.constant(operands[0]),)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    subscripts = "ii"
    operands = [np.random.rand(3, 3).astype(np.float32)]
    input_dict = {"subscripts": subscripts, "operands": (tf.constant(operands[0]),)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    subscripts = "ij,j->i"
    operands = [np.random.rand(4, 5).astype(np.float32), np.random.rand(5).astype(np.float32)]
    input_dict = {"subscripts": subscripts, "operands": (tf.constant(operands[0]), tf.constant(operands[1]))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    subscripts = "i,j->ij"
    operands = [np.random.rand(2).astype(np.float32), np.random.rand(3).astype(np.float32)]
    input_dict = {"subscripts": subscripts, "operands": (tf.constant(operands[0]), tf.constant(operands[1]))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    subscripts = "...,...->..."
    operands = [np.random.rand(2, 3, 4).astype(np.float32), np.random.rand(2, 3, 4).astype(np.float32)]
    input_dict = {"subscripts": subscripts, "operands": (tf.constant(operands[0]), tf.constant(operands[1]))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    subscripts = "i,i"
    operands = [np.random.rand(5).astype(np.float32), np.random.rand(5).astype(np.float32)]
    input_dict = {"subscripts": subscripts, "operands": (tf.constant(operands[0]), tf.constant(operands[1]))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    subscripts = "ij,ik->ijk"
    operands = [np.random.rand(2, 3).astype(np.float32), np.random.rand(2, 4).astype(np.float32)]
    input_dict = {"subscripts": subscripts, "operands": (tf.constant(operands[0]), tf.constant(operands[1]))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    subscripts = "i,j,k->ijk"
    operands = [np.random.rand(2).astype(np.float32), np.random.rand(3).astype(np.float32), np.random.rand(4).astype(np.float32)]
    input_dict = {"subscripts": subscripts, "operands": (tf.constant(operands[0]), tf.constant(operands[1]), tf.constant(operands[2]))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    subscripts = "abc,cd->abd"
    operands = [np.random.rand(2,3,4).astype(np.float32), np.random.rand(4,5).astype(np.float32)]
    input_dict = {"subscripts": subscripts, "operands": (tf.constant(operands[0]), tf.constant(operands[1]))}
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
