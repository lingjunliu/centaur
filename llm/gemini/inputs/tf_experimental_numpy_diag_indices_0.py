
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_experimental_numpy_diag_indices_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    n = 5
    ndim = 2
    input_dict = {"n": n, "ndim": ndim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different ndim
    n = 3
    ndim = 3
    input_dict = {"n": n, "ndim": ndim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Small n
    n = 2
    ndim = 2
    input_dict = {"n": n, "ndim": ndim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large n
    n = 10
    ndim = 2
    input_dict = {"n": n, "ndim": ndim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: ndim > n
    n = 3
    ndim = 5
    input_dict = {"n": n, "ndim": ndim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: ndim = 1
    n = 4
    ndim = 1
    input_dict = {"n": n, "ndim": ndim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger ndim
    n = 6
    ndim = 4
    input_dict = {"n": n, "ndim": ndim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: n = 1
    n = 1
    ndim = 2
    input_dict = {"n": n, "ndim": ndim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: n = ndim
    n = 4
    ndim = 4
    input_dict = {"n": n, "ndim": ndim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: ndim = 0 (should cause error)
    n = 4
    ndim = 0
    input_dict = {"n": n, "ndim": ndim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: larger n and ndim
    n = 8
    ndim = 3
    input_dict = {"n": n, "ndim": ndim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.diag_indices"] = tf_experimental_numpy_diag_indices_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.diag_indices' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.diag_indices'.")

check_valid('tf.experimental.numpy.diag_indices', generated_inputs['tf.experimental.numpy.diag_indices'], lib="tf", suffix=0)
