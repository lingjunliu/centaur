
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_diagflat_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, k=0
    v = tf.constant(np.array([1, 2, 3]))
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor, k=1
    v = tf.constant(np.array([4, 5, 6]))
    k = 1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor, k=-1
    v = tf.constant(np.array([7, 8, 9]))
    k = -1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D tensor, k=0
    v = tf.constant(np.array([[1, 2], [3, 4]]))
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D tensor, k=1
    v = tf.constant(np.array([[5, 6], [7, 8]]))
    k = 1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D tensor, k=-1
    v = tf.constant(np.array([[9, 10], [11, 12]]))
    k = -1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 1D tensor, k=2
    v = tf.constant(np.array([10, 11]))
    k = 2
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D tensor, k=-2
    v = tf.constant(np.array([12, 13]))
    k = -2
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: empty tensor
    v = tf.constant(np.array([]))
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D tensor, k=0
    v = tf.constant(np.array([[1, 2]]))
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs = {}
generated_inputs["tf.experimental.numpy.diagflat"] = tf_experimental_numpy_diagflat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.diagflat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.diagflat'.")

check_valid('tf.experimental.numpy.diagflat', generated_inputs['tf.experimental.numpy.diagflat'], lib="tf", suffix=0)
