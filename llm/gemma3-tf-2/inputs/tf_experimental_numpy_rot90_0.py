
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_rot90_inputs():
    list_of_inputs = []

    m1 = np.array([[1, 2], [3, 4]])
    k1 = 1
    axes1 = (0, 1)
    input_dict1 = {"m": m1, "k": k1, "axes": axes1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    m2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k2 = 2
    axes2 = (0, 1)
    input_dict2 = {"m": m2, "k": k2, "axes": axes2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    m3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k3 = -1
    axes3 = (0, 1)
    input_dict3 = {"m": m3, "k": k3, "axes": axes3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    m4 = np.array([[1, 2], [3, 4], [5, 6]])
    k4 = 1
    axes4 = (0, 1)
    input_dict4 = {"m": m4, "k": k4, "axes": axes4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    m5 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    k5 = 3
    axes5 = (0, 2)
    input_dict5 = {"m": m5, "k": k5, "axes": axes5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    m6 = np.array([[1, 2], [3, 4]])
    k6 = 0
    axes6 = (0, 1)
    input_dict6 = {"m": m6, "k": k6, "axes": axes6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    m7 = np.array([[[1, 2], [3, 4]]])
    k7 = 2
    axes7 = (1, 2)
    input_dict7 = {"m": m7, "k": k7, "axes": axes7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    m8 = np.array([[[1, 2, 3], [4, 5, 6]]])
    k8 = 1
    axes8 = (0, 1)
    input_dict8 = {"m": m8, "k": k8, "axes": axes8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    m9 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    k9 = 1
    axes9 = (0, 1)
    input_dict9 = {"m": m9, "k": k9, "axes": axes9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    m10 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    k10 = 3
    axes10 = (0, 1)
    input_dict10 = {"m": m10, "k": k10, "axes": axes10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.rot90"] = tf_rot90_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.rot90' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.rot90'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.rot90', generated_inputs['tf.experimental.numpy.rot90'], lib="tf", suffix=0)
