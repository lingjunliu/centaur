
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_diag_inputs():
    list_of_inputs = []

    # Input 1: 1D array, default k=0
    v = np.array([1, 2, 3])
    k = 0
    input_dict = {"v": tf.constant(v).numpy(), "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, k=1
    v = np.array([1, 2, 3])
    k = 1
    input_dict = {"v": tf.constant(v).numpy(), "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, k=-1
    v = np.array([1, 2, 3])
    k = -1
    input_dict = {"v": tf.constant(v).numpy(), "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, default k=0
    v = np.array([[1, 2], [3, 4]])
    k = 0
    input_dict = {"v": tf.constant(v).numpy(), "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, k=1
    v = np.array([[1, 2], [3, 4]])
    k = 1
    input_dict = {"v": tf.constant(v).numpy(), "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, k=-1
    v = np.array([[1, 2], [3, 4]])
    k = -1
    input_dict = {"v": tf.constant(v).numpy(), "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with float values
    v = np.array([1.0, 2.5, 3.2])
    k = 0
    input_dict = {"v": tf.constant(v).numpy(), "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with negative values
    v = np.array([-1, -2, -3])
    k = 0
    input_dict = {"v": tf.constant(v).numpy(), "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array with mixed values
    v = np.array([[1, -2], [3.5, 4]])
    k = 0
    input_dict = {"v": tf.constant(v).numpy(), "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, larger k
    v = np.array([1, 2, 3, 4, 5])
    k = 2
    input_dict = {"v": tf.constant(v).numpy(), "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.diag"] = tf_experimental_numpy_diag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.diag'.")

check_valid('tf.experimental.numpy.diag', generated_inputs['tf.experimental.numpy.diag'], lib="tf", suffix=0)
