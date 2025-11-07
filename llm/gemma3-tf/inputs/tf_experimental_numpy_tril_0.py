
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_tril_inputs():
    list_of_inputs = []

    m1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k1 = 0
    input_dict1 = {"m": m1, "k": k1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    m2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k2 = -1
    input_dict2 = {"m": m2, "k": k2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    m3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k3 = 1
    input_dict3 = {"m": m3, "k": k3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    m4 = np.array([[1, 2], [3, 4]])
    k4 = -1
    input_dict4 = {"m": m4, "k": k4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    m5 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    k5 = 2
    input_dict5 = {"m": m5, "k": k5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    m6 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k6 = 0
    input_dict6 = {"m": m6, "k": k6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    m7 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    k7 = -1
    input_dict7 = {"m": m7, "k": k7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    m8 = np.array([[1.0, 2.0], [3.0, 4.0]])
    k8 = 0
    input_dict8 = {"m": m8, "k": k8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    m9 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k9 = 3
    input_dict9 = {"m": m9, "k": k9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    m10 = np.array([[1, 2, 3, 4]])
    k10 = -2
    input_dict10 = {"m": m10, "k": k10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.tril"] = tf_experimental_numpy_tril_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.tril' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.tril'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.tril', generated_inputs['tf.experimental.numpy.tril'], lib="tf", suffix=0)
